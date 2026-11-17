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
    with open(os.path.join(OUT, 'index.json'), 'w') as f:
        json.dump(index, f)
    with open(os.path.join(OUT, 'index.html'), 'w') as f:
        f.write('<meta charset="utf-8"><p>Dati delle mappe di Meteo Center. ' + json.dumps({k: v['run'] for k, v in index['models'].items() if v}) + '</p>')


if __name__ == '__main__':
    main()
