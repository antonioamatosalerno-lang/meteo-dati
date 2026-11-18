#!/usr/bin/env python3
"""
Prepara i dati delle mappe di Meteo Center (gira da solo su GitHub ogni 3 ore).

Scarica i file originali dei modelli, gratuiti e pubblici:
  - ECMWF IFS 0,25°  (data.ecmwf.int, «open data»)
  - ICON-EU 0,0625°  (opendata.dwd.de, servizio meteo tedesco)
  - GFS 0,25°        (nomads.ncep.noaa.gov, servizio meteo americano, già ritagliato sull'Europa)
li ritaglia su due zone (Europa larga ogni 0,5° e Italia ogni 0,1°), ogni 3 ore fino a 5 giorni,
e scrive per ogni modello, zona e variabile un file compresso:  m/<modello>/<zona>/<variabile>.bin
(numeri interi a 16 bit compressi con gzip: valore = offset + scala × numero; -32768 = manca).
In index.json ci sono la corsa, le ore e come leggere i numeri. L'app scarica solo quello che serve.

Unità (come le usa l'app): temperature °C, altezze m, vento km/h, pioggia mm in 3 ore,
neve cm in 3 ore, nuvole %, pressione hPa.
"""
import bz2
import concurrent.futures as cf
import datetime as dt
import gzip
import json
import re
import os
import sys
import tempfile
import time
import traceback

import eccodes
import numpy as np
import requests

OUT = sys.argv[1] if len(sys.argv) > 1 else 'sito'
ONLY = sys.argv[2].split(',') if len(sys.argv) > 2 else ['ecmwf', 'icon', 'gfs']
SITE = 'https://antonioamatosalerno-lang.github.io/meteo-dati/'
STEPS = list(range(0, 121, 3))
LEVELS = [850, 700, 500, 250]
DOMAINS = {
    'eu': dict(w=-45.0, e=60.0, s=22.0, n=75.0, step=0.5),
    'it': dict(w=5.0, e=20.0, s=35.0, n=48.0, step=0.1),
}
for d in DOMAINS.values():
    d['nx'] = int(round((d['e'] - d['w']) / d['step'])) + 1
    d['ny'] = int(round((d['n'] - d['s']) / d['step'])) + 1

# scala per i numeri interi (e offset): precisione e intervallo che bastano
QUANT = {'t': (0.01, 0), 'z': (1.0, 0), 'u': (0.02, 0), 'v': (0.02, 0), 'tp': (0.01, 0), 'sf': (0.01, 0),
         'tcc': (0.01, 0), 'gust': (0.01, 0), 'msl': (0.01, 1000), 'zero': (1.0, 0)}
def quant(var):
    for k in ('tp', 'sf', 'tcc', 'gust', 'msl', 'zero'):
        if var == k:
            return QUANT[k]
    return QUANT[var[0]] if var[0] in 'tzuv' else (0.01, 0)

ALL_VARS = ['t2m', 'u10', 'v10', 'msl', 'tp', 'sf', 'tcc', 'gust', 'zero'] + \
    [f'{p}{l}' for l in LEVELS for p in ('t', 'z', 'u', 'v')]

PREV = {}  # corse già pubblicate la volta prima: modello → 'AAAA-MM-GGTHH:00Z'


class SameRun(Exception):
    pass


def same(name, run):
    if PREV.get(name) == run.strftime('%Y-%m-%dT%H:00Z'):
        raise SameRun()


S = requests.Session()
S.headers['User-Agent'] = 'MeteoCenter-mappe (uso personale)'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def get(url, tries=4, **kw):
    for k in range(tries):
        try:
            r = S.get(url, timeout=120, **kw)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(3 * (k + 1))
    return None


def head_ok(url):
    try:
        return S.head(url, timeout=30, allow_redirects=True).status_code == 200
    except requests.RequestException:
        return False


# ====================================================================== lettura GRIB e ritaglio

def messages(data):
    """I campi contenuti in un file GRIB (bytes)."""
    with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as f:
        f.write(data)
        name = f.name
    out = []
    try:
        with open(name, 'rb') as f:
            while True:
                h = eccodes.codes_grib_new_from_file(f)
                if h is None:
                    break
                try:
                    g = lambda k: eccodes.codes_get(h, k)
                    ni, nj = g('Ni'), g('Nj')
                    vals = eccodes.codes_get_values(h).astype(np.float32)
                    if eccodes.codes_get(h, 'bitmapPresent'):
                        vals[vals == g('missingValue')] = np.nan
                    vals = vals.reshape(nj, ni)
                    la1, la2 = g('latitudeOfFirstGridPointInDegrees'), g('latitudeOfLastGridPointInDegrees')
                    lo1, lo2 = g('longitudeOfFirstGridPointInDegrees'), g('longitudeOfLastGridPointInDegrees')
                    if lo2 < lo1:
                        lo2 += 360
                    out.append(dict(short=g('shortName'), level=g('level'), tlev=g('typeOfLevel'),
                                    start=g('startStep'), end=g('endStep'), vals=vals,
                                    lats=np.linspace(la1, la2, nj), lons=np.linspace(lo1, lo2, ni)))
                finally:
                    eccodes.codes_release(h)
    finally:
        os.unlink(name)
    return out


_grid_cache = {}
def regrid(m, dom):
    """Il campo [m] sulla griglia della zona [dom] (interpolazione bilineare)."""
    D = DOMAINS[dom]
    lats, lons, v = m['lats'], m['lons'], m['vals']
    lo = ((lons + 180) % 360) - 180
    oj = np.argsort(lo, kind='stable')
    oi = np.argsort(lats, kind='stable')
    key = (dom, len(lats), float(lats[0]), float(lats[-1]), len(lons), float(lons[0]), float(lons[-1]))
    if key not in _grid_cache:
        la, lo2 = lats[oi], lo[oj]
        tl = D['s'] + np.arange(D['ny']) * D['step']
        tg = D['w'] + np.arange(D['nx']) * D['step']
        fi = np.interp(tl, la, np.arange(len(la)), left=np.nan, right=np.nan)
        fj = np.interp(tg, lo2, np.arange(len(lo2)), left=np.nan, right=np.nan)
        _grid_cache[key] = (fi, fj)
    fi, fj = _grid_cache[key]
    v = v[oi][:, oj]
    okI, okJ = ~np.isnan(fi), ~np.isnan(fj)
    i0 = np.clip(np.floor(np.nan_to_num(fi)).astype(int), 0, v.shape[0] - 2)
    j0 = np.clip(np.floor(np.nan_to_num(fj)).astype(int), 0, v.shape[1] - 2)
    di = (np.nan_to_num(fi) - i0)[:, None]
    dj = (np.nan_to_num(fj) - j0)[None, :]
    a = v[np.ix_(i0, j0)]; b = v[np.ix_(i0, j0 + 1)]; c = v[np.ix_(i0 + 1, j0)]; d = v[np.ix_(i0 + 1, j0 + 1)]
    r = a * (1 - di) * (1 - dj) + b * (1 - di) * dj + c * di * (1 - dj) + d * di * dj
    r[~okI, :] = np.nan
    r[:, ~okJ] = np.nan
    return r.astype(np.float32)


class Store:
    """Raccoglie i campi di un modello: var → zona → array (ore, ny, nx)."""
    def __init__(self):
        self.data = {}

    def put(self, var, step, m, conv=lambda x: x):
        if step not in STEPS:
            return
        k = STEPS.index(step)
        for dom, D in DOMAINS.items():
            a = self.data.setdefault(var, {}).setdefault(dom, np.full((len(STEPS), D['ny'], D['nx']), np.nan, np.float32))
            a[k] = conv(regrid(m, dom))

    def get(self, var, dom):
        return self.data.get(var, {}).get(dom)

    def set(self, var, dom, arr):
        self.data.setdefault(var, {})[dom] = arr.astype(np.float32)


def deaccumulate(st, var, scale):
    """Da «caduta dall'inizio» a «caduta nelle ultime 3 ore» (× scale)."""
    for dom in DOMAINS:
        a = st.get(var, dom)
        if a is None:
            continue
        b = np.zeros_like(a)
        b[1:] = np.maximum(0, a[1:] - a[:-1]) * scale
        st.set(var, dom, np.nan_to_num(b))


def freezing_from_levels(st):
    """Zero termico stimato dai livelli 850/700/500 (per chi non lo dà): dove la temperatura passa per 0 °C."""
    for dom in DOMAINS:
        t8, t7, t5 = (st.get(f't{l}', dom) for l in (850, 700, 500))
        z8, z7, z5 = (st.get(f'z{l}', dom) for l in (850, 700, 500))
        if any(x is None for x in (t8, t7, t5, z8, z7, z5)):
            continue
        out = np.where(t8 <= 0, z8 + t8 / 0.0065, np.nan)                       # sotto 850: si scende con 6,5 °C/km
        out = np.where((t8 > 0) & (t7 <= 0), z8 + (z7 - z8) * t8 / (t8 - t7 + 1e-6), out)
        out = np.where((t7 > 0) & (t5 <= 0), z7 + (z5 - z7) * t7 / (t7 - t5 + 1e-6), out)
        out = np.where(t5 > 0, z5, out)
        st.set('zero', dom, np.maximum(out, 0))


# ====================================================================== ECMWF

def ecmwf():
    from ecmwf.opendata import Client
    c = Client(source='ecmwf', model='ifs', resol='0p25')
    run = c.latest(type='fc', step=120, param='2t')
    log('ECMWF corsa', run)
    same('ecmwf', run)
    st = Store()
    tmp = tempfile.mkdtemp()

    def fetch(name, **kw):
        path = os.path.join(tmp, name + '.grib2')
        for k in range(3):
            try:
                c.retrieve(date=run.strftime('%Y%m%d'), time=run.hour, type='fc', step=STEPS, target=path, **kw)
                with open(path, 'rb') as f:
                    return f.read()
            except Exception as e:  # noqa
                log('ECMWF', name, 'riprovo:', e)
                time.sleep(10)
        return None

    sfc = {'10u': ('u10', lambda x: x * 3.6), '10v': ('v10', lambda x: x * 3.6), '2t': ('t2m', lambda x: x - 273.15),
           'msl': ('msl', lambda x: x / 100), 'tp': ('tp', None), 'sf': ('sf', None), 'tcc': ('tcc', None),
           '10fg': ('gust', lambda x: x * 3.6)}
    for p, (var, conv) in sfc.items():
        data = fetch(p, param=p)
        if not data:
            log('ECMWF manca', p)
            continue
        for m in messages(data):
            st.put(var, m['end'], m, conv or (lambda x: x))
    if st.get('tcc', 'eu') is not None and np.nanmax(st.get('tcc', 'eu')) <= 1.5:
        for dom in DOMAINS:
            st.set('tcc', dom, st.get('tcc', dom) * 100)
    deaccumulate(st, 'tp', 1000)
    deaccumulate(st, 'sf', 1000)      # metri d'acqua → circa 1 cm di neve per mm d'acqua
    data = fetch('pl', levtype='pl', levelist=LEVELS, param=['u', 'v', 't', 'gh'])
    for m in messages(data or b''):
        conv = {'u': lambda x: x * 3.6, 'v': lambda x: x * 3.6, 't': lambda x: x - 273.15, 'gh': lambda x: x}[m['short']]
        var = {'gh': 'z'}.get(m['short'], m['short']) + str(m['level'])
        st.put(var, m['end'], m, conv)
    freezing_from_levels(st)
    return run, st


# ====================================================================== ICON-EU (DWD)

def icon():
    now = dt.datetime.utcnow()
    run = None
    for h in range(3, 40):
        r = (now - dt.timedelta(hours=h)).replace(minute=0, second=0, microsecond=0)
        if r.hour % 6:
            continue
        url = icon_url(r, 120, 't_2m')
        if head_ok(url):
            run = r
            break
    if run is None:
        raise RuntimeError('nessuna corsa ICON-EU completa')
    log('ICON corsa', run)
    same('icon', run)
    st = Store()
    single = {'t_2m': ('t2m', lambda x: x - 273.15), 'u_10m': ('u10', lambda x: x * 3.6), 'v_10m': ('v10', lambda x: x * 3.6),
              'pmsl': ('msl', lambda x: x / 100), 'tot_prec': ('tp', None), 'clct': ('tcc', None),
              'vmax_10m': ('gust', lambda x: x * 3.6), 'hzerocl': ('zero', None),
              'snow_gsp': ('sfa', None), 'snow_con': ('sfb', None)}
    jobs = [(v, None, s) for v in single for s in STEPS] + \
        [(v, l, s) for v in ('t', 'u', 'v', 'fi') for l in LEVELS for s in STEPS]

    def one(job):
        v, l, s = job
        data = get(icon_url(run, s, v, l))
        if not data:
            return job, []
        return job, messages(bz2.decompress(data))

    with cf.ThreadPoolExecutor(12) as ex:
        for (v, l, s), msgs in ex.map(one, jobs):
            for m in msgs:
                if l is None:
                    var, conv = single[v]
                else:
                    var = {'fi': 'z'}.get(v, v) + str(l)
                    conv = {'t': lambda x: x - 273.15, 'u': lambda x: x * 3.6, 'v': lambda x: x * 3.6, 'fi': lambda x: x / 9.80665}[v]
                st.put(var, s, m, conv or (lambda x: x))
    deaccumulate(st, 'tp', 1)
    deaccumulate(st, 'sfa', 1)
    deaccumulate(st, 'sfb', 1)
    for dom in DOMAINS:
        a, b = st.get('sfa', dom), st.get('sfb', dom)
        if a is not None:
            st.set('sf', dom, a + (b if b is not None else 0))
    st.data.pop('sfa', None)
    st.data.pop('sfb', None)
    for dom in DOMAINS:  # zero termico: -999 = sotto il suolo
        z = st.get('zero', dom)
        if z is not None:
            st.set('zero', dom, np.where(z < 0, 0, z))
    return run, st


def icon_url(run, step, var, level=None):
    base = f'https://opendata.dwd.de/weather/nwp/icon-eu/grib/{run:%H}/{var}/'
    if level is None:
        return base + f'icon-eu_europe_regular-lat-lon_single-level_{run:%Y%m%d%H}_{step:03d}_{var.upper()}.grib2.bz2'
    return base + f'icon-eu_europe_regular-lat-lon_pressure-level_{run:%Y%m%d%H}_{step:03d}_{level}_{var.upper()}.grib2.bz2'


# ====================================================================== GFS (NOAA)

def gfs():
    now = dt.datetime.utcnow()
    run = None
    for h in range(4, 40):
        r = (now - dt.timedelta(hours=h)).replace(minute=0, second=0, microsecond=0)
        if r.hour % 6:
            continue
        if head_ok(f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{r:%Y%m%d}/{r:%H}/atmos/gfs.t{r:%H}z.pgrb2.0p25.f120.idx'):
            run = r
            break
    if run is None:
        raise RuntimeError('nessuna corsa GFS completa')
    log('GFS corsa', run)
    same('gfs', run)
    st = Store()
    var = '&'.join(f'var_{v}=on' for v in ('TMP', 'UGRD', 'VGRD', 'HGT', 'PRMSL', 'APCP', 'CSNOW', 'TCDC', 'GUST'))
    lev = '&'.join(f'lev_{l}=on' for l in ('2_m_above_ground', '10_m_above_ground', 'mean_sea_level', 'surface', 'entire_atmosphere',
                                           '0C_isotherm') + tuple(f'{p}_mb' for p in LEVELS))
    apcp, csnow = {}, {}

    def one(s):
        url = (f'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?dir=%2Fgfs.{run:%Y%m%d}%2F{run:%H}%2Fatmos'
               f'&file=gfs.t{run:%H}z.pgrb2.0p25.f{s:03d}&{var}&{lev}&subregion=&toplat=76&leftlon=-46&rightlon=61&bottomlat=21')
        data = get(url)
        return s, messages(data) if data else []

    with cf.ThreadPoolExecutor(3) as ex:   # NOMADS vuole poche richieste insieme
        for s, msgs in ex.map(one, STEPS):
            for m in msgs:
                sh, tl, lv = m['short'], m['tlev'], m['level']
                if tl == 'heightAboveGround' and lv == 2 and sh == '2t':
                    st.put('t2m', s, m, lambda x: x - 273.15)
                elif tl == 'heightAboveGround' and lv == 10 and sh in ('10u', '10v'):
                    st.put(sh[2] + '10', s, m, lambda x: x * 3.6)
                elif sh == 'prmsl':
                    st.put('msl', s, m, lambda x: x / 100)
                elif sh == 'tp':
                    apcp[s] = (m['end'] - m['start'], {d: regrid(m, d) for d in DOMAINS})
                elif sh == 'csnow':
                    csnow[s] = {d: regrid(m, d) for d in DOMAINS}
                elif sh == 'tcc' and tl == 'atmosphere':
                    st.put('tcc', s, m)
                elif sh == 'gust' and tl == 'surface':
                    st.put('gust', s, m, lambda x: x * 3.6)
                elif sh == 'gh' and tl == 'isothermZero':
                    st.put('zero', s, m)
                elif tl == 'isobaricInhPa' and lv in LEVELS and sh in ('t', 'u', 'v', 'gh'):
                    conv = {'t': lambda x: x - 273.15, 'u': lambda x: x * 3.6, 'v': lambda x: x * 3.6, 'gh': lambda x: x}[sh]
                    st.put({'gh': 'z'}.get(sh, sh) + str(lv), s, m, conv)
    # pioggia: GFS dà secchi di 3 o 6 ore
    for dom, D in DOMAINS.items():
        tp = np.zeros((len(STEPS), D['ny'], D['nx']), np.float32)
        sf = np.zeros_like(tp)
        for k, s in enumerate(STEPS):
            if s not in apcp:
                continue
            span, g = apcp[s]
            v = g[dom]
            if span == 6 and (s - 3) in apcp:
                v = v - apcp[s - 3][1][dom]
            tp[k] = np.maximum(np.nan_to_num(v), 0)
            if s in csnow:
                sf[k] = np.where(np.nan_to_num(csnow[s][dom]) > 0.5, tp[k], 0)
        st.set('tp', dom, tp)
        st.set('sf', dom, sf)
    return run, st


# ====================================================================== scrittura

def write(name, run, st):
    times = [int((run + dt.timedelta(hours=s)).replace(tzinfo=dt.timezone.utc).timestamp() * 1000) for s in STEPS]
    info = {'run': run.strftime('%Y-%m-%dT%H:00Z'), 'times': times, 'vars': {}}
    for var in ALL_VARS:
        sc, off = quant(var)
        okall = True
        for dom in DOMAINS:
            a = st.get(var, dom)
            if a is None or np.all(np.isnan(a)):
                okall = False
                break
        if not okall:
            log(name, 'senza', var)
            continue
        for dom in DOMAINS:
            a = st.get(var, dom)
            q = np.round((a - off) / sc)
            q = np.where(np.isnan(q), -32768, np.clip(q, -32767, 32767)).astype('<i2')
            path = os.path.join(OUT, 'm', name, dom)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, var + '.bin'), 'wb') as f:
                f.write(gzip.compress(q.tobytes(), 6))
        info['vars'][var] = {'s': sc, 'o': off}
    return info


def keep_old(name, old):
    """Se un modello non riesce, si tengono i file della volta prima (scaricati dal sito)."""
    m = (old or {}).get('models', {}).get(name)
    if not m:
        return None
    for var in m['vars']:
        for dom in DOMAINS:
            data = get(f'{SITE}m/{name}/{dom}/{var}.bin')
            if data:
                path = os.path.join(OUT, 'm', name, dom)
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, var + '.bin'), 'wb') as f:
                    f.write(data)
    log(name, 'tengo la corsa precedente', m['run'])
    return m


# ====================================================================== allerte della Protezione Civile

DPC = 'pcm-dpc/DPC-Bollettini-Criticita-Idrogeologica-Idraulica'


def allerte():
    """L'ultimo bollettino di criticità della Protezione Civile (zone d'allerta di oggi e domani), alleggerito.
    Dati aperti CC-BY 4.0: https://github.com/pcm-dpc/DPC-Bollettini-Criticita-Idrogeologica-Idraulica"""
    h = {'Accept': 'application/vnd.github+json'}
    if os.environ.get('GITHUB_TOKEN'):
        h['Authorization'] = 'Bearer ' + os.environ['GITHUB_TOKEN']
    # l'elenco completo dei bollettini (cartella files/geojson) e si prende il più recente
    def tree(sha):
        return S.get(f'https://api.github.com/repos/{DPC}/git/trees/{sha}', headers=h, timeout=60).json()['tree']
    files_sha = next(x['sha'] for x in tree('master') if x['path'] == 'files')
    geo_sha = next(x['sha'] for x in tree(files_sha) if x['path'] == 'geojson')
    names = sorted(x['path'] for x in tree(geo_sha) if x['path'].endswith('_today.json'))
    if not names:
        raise RuntimeError('bollettino non trovato')
    stem = names[-1][:-len('_today.json')]          # es. 20261117_1519
    day = dt.datetime.strptime(stem[:8], '%Y%m%d').date()

    def level(s):
        s = (s or '').upper()
        return 3 if 'ROSSA' in s else 2 if 'ARANCIONE' in s else 1 if 'GIALLA' in s else 0

    def rnd(c):
        return [rnd(x) for x in c] if isinstance(c, list) and c and isinstance(c[0], list) else [round(c[0], 3), round(c[1], 3)]

    out = {'issued': stem, 'source': 'Protezione Civile - Bollettini di criticità e di vigilanza meteo (CC-BY 4.0); '
                                     'Aeronautica Militare via Meteoalarm', 'days': {}}
    geom = None
    for part, d in (('today', day), ('tomorrow', day + dt.timedelta(days=1))):
        g = json.loads(get(f'https://raw.githubusercontent.com/{DPC}/master/files/geojson/{stem}_{part}.json'))
        zones = []
        for ft in g['features']:
            p = ft['properties']
            geo = ft['geometry']
            zones.append({'zona': p.get('Nome zona', ''),
                          'temporali': level(p.get('Per rischio temporali')),
                          'idrogeologico': level(p.get('Per rischio idrogeologico')),
                          'idraulico': level(p.get('Per rischio idraulico')),
                          'type': geo['type'], 'coords': rnd(geo['coordinates'])})
        out['days'][d.isoformat()] = zones
        geom = geom or zones
    # oggi e domani ci sono sempre (anche prima che esca il bollettino di criticità del giorno): zone senza criticità
    oggi = dt.datetime.now(ROME).date()
    for d in (oggi, oggi + dt.timedelta(days=1)):
        if d.isoformat() not in out['days']:
            out['days'][d.isoformat()] = [dict(z, temporali=0, idrogeologico=0, idraulico=0) for z in geom]
    for k in [k for k in out['days'] if k < oggi.isoformat()]:
        del out['days'][k]
    # gli altri bollettini, zona per zona
    for name, fn in (('vigilanza', vigilanza), ('meteoalarm', meteoalarm)):
        try:
            fn(out['days'])
        except Exception:
            traceback.print_exc()
            log('allerte:', name, 'non riuscito')
    # i testi: della vigilanza per giorno, degli avvisi in un elenco (le zone li indicano col numero)
    out['vt'] = {k: v for k, v in VIG_TESTI.items() if k in out['days']}
    out['t'] = TESTI
    with open(os.path.join(OUT, 'allerte.json'), 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    log('allerte: bollettino', stem, 'giorni', list(out['days']))


# ---------------------------------------------------------------------- strumenti per le zone

from zoneinfo import ZoneInfo
ROME = ZoneInfo('Europe/Rome')


def polys(typ, c):
    """Elenco di poligoni (ognuno: lista di anelli [[lon, lat], ...]) di una geometria."""
    return [c] if typ == 'Polygon' else list(c)


def inside(pt, pl):
    """Il punto è dentro la geometria (lista di poligoni con eventuali buchi)?"""
    x, y = pt
    for poly in pl:
        n = 0
        for ring in poly:
            inr = False
            j = len(ring) - 1
            for i in range(len(ring)):
                xi, yi = ring[i]
                xj, yj = ring[j]
                if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                    inr = not inr
                j = i
            n += inr
        if n % 2 == 1:
            return True
    return False


def centro(pl):
    """Un punto dentro la zona (il centro del poligono più grande, o un suo vertice vicino)."""
    big = max(pl, key=lambda p: len(p[0]))[0]
    cx = sum(p[0] for p in big) / len(big)
    cy = sum(p[1] for p in big) / len(big)
    if inside((cx, cy), pl):
        return cx, cy
    return min(big, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)


def km(a, b):
    import math
    dx = (a[0] - b[0]) * 111.3 * math.cos(math.radians((a[1] + b[1]) / 2))
    return math.hypot(dx, (a[1] - b[1]) * 111.3)


def topo(tj):
    """TopoJSON → [(proprietà, lista di poligoni)]."""
    tr = tj.get('transform')
    arcs = []
    for a in tj['arcs']:
        if tr:
            x = y = 0
            pts = []
            for dx, dy in a:
                x += dx
                y += dy
                pts.append([x * tr['scale'][0] + tr['translate'][0], y * tr['scale'][1] + tr['translate'][1]])
        else:
            pts = [list(p[:2]) for p in a]
        arcs.append(pts)

    def ring(ids):
        r = []
        for i in ids:
            p = arcs[i] if i >= 0 else arcs[~i][::-1]
            r += p if not r else p[1:]
        return r
    out = []
    for obj in tj['objects'].values():
        for g in obj['geometries']:
            if g['type'] == 'Polygon':
                out.append((g.get('properties', {}), [[ring(r) for r in g['arcs']]]))
            elif g['type'] == 'MultiPolygon':
                out.append((g.get('properties', {}), [[ring(r) for r in p] for p in g['arcs']]))
            elif g['type'] == 'Point':
                out.append((g.get('properties', {}), None))
    return out


# ---------------------------------------------------------------------- bollettino di vigilanza meteo (Protezione Civile)

VIG = 'https://raw.githubusercontent.com/pcm-dpc/DPC-Bollettini-Vigilanza-Meteorologica/master/files/'
# i simboli della mappa (codici ricavati confrontando simboli e testo del bollettino)
FENOMENI = {1: 'piogge sparse', 2: 'piogge diffuse', 3: 'nevicate', 4: 'nevicate abbondanti',
            5: 'rovesci o temporali', 6: 'rovesci o temporali', 7: 'rovesci o temporali diffusi',
            10: 'venti forti', 11: 'venti di burrasca', 12: 'venti di tempesta', 13: 'forti raffiche di vento',
            40: 'mare molto mosso', 41: 'mare agitato', 42: 'mare grosso'}
PIOGGIA = {3: 'piogge moderate', 4: 'piogge abbondanti', 5: 'piogge molto abbondanti'}
# per ogni tipo di fenomeno si tiene solo il più forte: (tipo, forza)
FORZA = {'piogge sparse': ('p', 1), 'piogge diffuse': ('p', 2), 'piogge moderate': ('p', 3), 'piogge abbondanti': ('p', 4),
         'piogge molto abbondanti': ('p', 5), 'rovesci o temporali': ('t', 1), 'rovesci o temporali diffusi': ('t', 2),
         'nevicate': ('n', 1), 'nevicate abbondanti': ('n', 2), 'forti raffiche di vento': ('v', 0), 'venti forti': ('v', 1),
         'venti di burrasca': ('v', 2), 'venti di tempesta': ('v', 3),
         'mare molto mosso': ('m', 1), 'mare agitato': ('m', 2), 'mare grosso': ('m', 3)}


def piu_forti(vg):
    best = {}
    for n in vg:
        k, f = FORZA.get(n, (n, 0))
        if k not in best or f > best[k][1]:
            best[k] = (n, f)
    ordine = 'tvpnm'
    return [n for k, (n, f) in sorted(best.items(), key=lambda x: ordine.find(x[0]) if x[0] in ordine else 9)]


VIG_TESTI = {}   # giorno → {argomento: testo}


def vig_testo(html):
    """«<p><strong>Venti:</strong> forti …</p>…» → {'Venti': 'forti …', …}"""
    import html as H
    import re
    out = {}
    cur = None
    for p in re.findall(r'<p[^>]*>(.*?)</p>', html, flags=re.S):
        t = H.unescape(re.sub(r'<[^>]+>', ' ', p))
        t = re.sub(r'\s+', ' ', t).strip()
        if not t:
            continue
        m = re.match(r'([A-Za-zÀ-ÿ ]{3,20}?)\s*:\s*(.*)', t)
        if m:
            cur = m.group(1).strip().capitalize()
            out[cur] = m.group(2).strip()
        elif cur:   # il testo continua nel paragrafo dopo (es. un elenco con i trattini)
            out[cur] = (out[cur] + '\n' + t).strip()
    return {k: v for k, v in out.items() if v}


def vigilanza(days):
    """Per ogni zona e giorno: i fenomeni segnalati dal bollettino di vigilanza meteo nazionale (campo «vg»)."""
    oggi = dt.datetime.now(ROME).date()
    idx = None
    for d in (oggi, oggi - dt.timedelta(days=1)):
        try:
            idx = json.loads(get(VIG + d.strftime('%Y%m%d') + '.json'))
            emesso = d
            break
        except Exception:
            continue
    if not idx:
        raise RuntimeError('bollettino di vigilanza non trovato')
    parti = [k for k in ('today', 'tomorrow', 'aftertomorrow') if isinstance(idx.get(k), dict)]
    for n, k in enumerate(parti):
        giorno = (emesso + dt.timedelta(days=n)).isoformat()
        if giorno not in days:
            continue
        # il testo del bollettino, diviso per argomento (Precipitazioni, Venti, Mari…)
        testo = vig_testo(idx[k].get('html_description') or '')
        if testo:
            VIG_TESTI[giorno] = testo
        files = idx[k].get('topo_json') or []
        zone_f = next((f for f in files if 'fenomeni' not in f), None)
        fen_f = next((f for f in files if 'fenomeni' in f), None)
        vz = topo(json.loads(get(zone_f))) if zone_f else []
        pts = [(p.get('id_fenomeno'), (p['lon'], p['lat'])) for p, _ in topo(json.loads(get(fen_f)))] if fen_f else []
        for z in days[giorno]:
            pl = polys(z['type'], z['coords'])
            c = centro(pl)
            vg = []
            for p, g in vz:
                if g and inside(c, g) and PIOGGIA.get(p.get('id_classificazione')):
                    vg.append(PIOGGIA[p['id_classificazione']])
                    break
            verts = [v for poly in pl for v in poly[0][::4]]
            x0, x1 = min(v[0] for v in verts) - 0.7, max(v[0] for v in verts) + 0.7
            y0, y1 = min(v[1] for v in verts) - 0.5, max(v[1] for v in verts) + 0.5
            for code, pt in pts:
                nome = FENOMENI.get(code)
                if not nome or nome in vg or not (x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1):
                    continue
                if inside(pt, pl) or min(km(pt, v) for v in verts) < 35:
                    vg.append(nome)
            if vg:
                z['vg'] = piu_forti(vg)
    log('vigilanza: bollettino del', emesso, 'giorni', parti)


# ---------------------------------------------------------------------- avvisi dell'Aeronautica Militare (Meteoalarm)

REGIONI = 'https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson'
TIPI = {1: 'vento', 2: 'neve o ghiaccio', 3: 'temporali', 4: 'nebbia', 5: 'caldo', 6: 'freddo', 7: 'mareggiate',
        8: 'incendi', 9: 'valanghe', 10: 'pioggia', 12: 'alluvioni', 13: 'pioggia e alluvioni'}


TESTI = []   # i testi degli avvisi, una volta sola ciascuno


def testo_n(s):
    """Il numero del testo nell'elenco TESTI (lo aggiunge se manca); -1 se vuoto."""
    if not s:
        return -1
    if s not in TESTI:
        TESTI.append(s)
    return TESTI.index(s)


def meteoalarm(days):
    """Per ogni zona e giorno: gli avvisi gialli, arancioni e rossi della sua regione (campo «av»: tipo, livello, dalle, alle)."""
    norm = lambda s: ''.join(ch for ch in s.lower() if ch.isalpha())[:6]
    reg = [(norm(f['properties']['reg_name']), polys(f['geometry']['type'], f['geometry']['coordinates']))
           for f in json.loads(get(REGIONI))['features']]
    feed = json.loads(get('https://feeds.meteoalarm.org/api/v1/warnings/feeds-italy'))['warnings']
    superati = set()
    for w in feed:
        for ref in (w['alert'].get('references') or '').split():
            parts = ref.split(',')
            if len(parts) >= 2:
                superati.add(parts[1])
    avvisi = {}   # regione → [(tipo, livello, inizio, fine)]
    for w in feed:
        a = w['alert']
        if a['identifier'] in superati or a.get('msgType') == 'Cancel':
            continue
        for i in a.get('info', []):
            if not i.get('language', '').startswith('it'):
                continue
            par = {p['valueName']: p['value'] for p in i.get('parameter', [])}
            try:
                lv = int(par['awareness_level'].split(';')[0]) - 1      # 2 giallo → 1, 3 arancione → 2, 4 rosso → 3
                tipo = TIPI.get(int(par['awareness_type'].split(';')[0]))
            except (KeyError, ValueError):
                continue
            if lv < 1 or not tipo:
                continue
            t0 = dt.datetime.fromisoformat(i.get('onset') or i['effective']).astimezone(ROME)
            t1 = dt.datetime.fromisoformat(i['expires']).astimezone(ROME)
            # il testo: descrizione (senza l'avvertenza finale) e cosa fare
            desc = (i.get('description') or '').split('(DISCLAIMER')[0].strip()
            # in fondo le zone in maiuscolo («… intensità APPENNINO»): in minuscolo, tra parentesi
            desc = re.sub(r"\s+([A-ZÀ-Ý][A-ZÀ-Ý ,']{2,})\.?$", lambda m: ' (zone: ' + m.group(1).strip().lower() + ').', desc)
            if desc and not desc.endswith('.'):
                desc += '.'
            istr = (i.get('instruction') or '').strip()
            # la prima frase arriva tutta in maiuscolo («STARE (RIMANERE) ATTENTI.»): in minuscolo, il resto com'è
            istr = re.sub(r"^([^a-z.]{4,})\.", lambda m: m.group(1).capitalize().replace(' (rimanere)', '') + '.', istr)
            for ar in i.get('area', []):
                avvisi.setdefault(norm(ar['areaDesc']), []).append((tipo, lv, t0, t1, testo_n(desc), testo_n(istr)))
    for giorno, zones in days.items():
        d0 = dt.datetime.fromisoformat(giorno).replace(tzinfo=ROME)
        d1 = d0 + dt.timedelta(days=1)
        for z in zones:
            c = centro(polys(z['type'], z['coords']))
            r = next((n for n, pl in reg if inside(c, pl)), None)
            best = {}
            for tipo, lv, t0, t1, td, ti in avvisi.get(r, []):
                if t1 <= d0 or t0 >= d1:
                    continue
                a, b = max(t0, d0), min(t1, d1)
                if tipo not in best or lv > best[tipo][1]:
                    best[tipo] = [tipo, lv, a.strftime('%H:%M'), '24:00' if b >= d1 else b.strftime('%H:%M'), td, ti]
            if best:
                z['av'] = sorted(best.values(), key=lambda x: -x[1])
    log('meteoalarm: regioni con avvisi', sorted(avvisi))


def main():
    os.makedirs(OUT, exist_ok=True)
    try:
        old = json.loads(get(SITE + 'index.json') or b'{}')
    except ValueError:
        old = {}
    same_zones = json.dumps(old.get('domains'), sort_keys=True) == json.dumps(DOMAINS, sort_keys=True)
    for k, v in (old.get('models') or {}).items():
        if v and same_zones:   # se le zone sono cambiate si rifà tutto
            PREV[k] = v['run']
    if not same_zones:
        old = {}
    index = {'made': int(time.time() * 1000), 'domains': DOMAINS, 'models': {}}
    for name, fn in (('ecmwf', ecmwf), ('icon', icon), ('gfs', gfs)):
        t0 = time.time()
        try:
            if name not in ONLY:
                raise RuntimeError('saltato')
            run, st = fn()
            index['models'][name] = write(name, run, st)
            log(name, 'fatto in', int(time.time() - t0), 's')
        except SameRun:
            log(name, 'corsa già pubblicata: la tengo')
            index['models'][name] = keep_old(name, old)
        except Exception:
            traceback.print_exc()
            m = keep_old(name, old)
            if m:
                index['models'][name] = m
    try:
        allerte()
    except Exception:
        traceback.print_exc()
        data = get(SITE + 'allerte.json')      # se non va, si tiene quello di prima
        if data:
            with open(os.path.join(OUT, 'allerte.json'), 'wb') as f:
                f.write(data)
    with open(os.path.join(OUT, 'index.json'), 'w') as f:
        json.dump(index, f)
    with open(os.path.join(OUT, 'index.html'), 'w') as f:
        f.write('<meta charset="utf-8"><p>Dati delle mappe di Meteo Center. ' + json.dumps({k: v['run'] for k, v in index['models'].items() if v}) + '</p>')


if __name__ == '__main__':
    main()
