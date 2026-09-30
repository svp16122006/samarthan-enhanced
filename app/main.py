from __future__ import annotations

import json, math, sqlite3, uuid, wave, struct, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.ml_core import load_model

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data' / 'samarthan.db'
MODEL = ROOT / 'models' / 'model.pkl'
METRICS = ROOT / 'models' / 'metrics.json'
UPLOADS = ROOT / 'data' / 'uploads'
UPLOADS.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(exist_ok=True)

app = FastAPI(title='Project Samarthan', version='0.8.0')
app.mount('/static', StaticFiles(directory=ROOT / 'app' / 'static'), name='static')
templates = Jinja2Templates(directory=str(ROOT / 'app' / 'templates'))

RISK_ORDER = {'stable': 0, 'elevated': 1, 'high': 2, 'critical': 3}
RISK_SCORE = {'stable': 18, 'elevated': 42, 'high': 68, 'critical': 88}
LANGUAGES = ['English', 'Hindi', 'Marathi', 'Bengali', 'Tamil', 'Telugu', 'Kannada', 'Malayalam', 'Gujarati', 'Punjabi', 'Odia', 'Assamese']

# Transparent engineering sanity layer for obvious positive/urgent language.
# This reduces obvious false positives from a small synthetic model; it is not clinical AI.
POSITIVE_PATTERNS = [
    r'\bfeel safe\b', r'\bextremely safe\b', r'\bsafe (today|at home|right now)\b', r'\bfeel supported\b',
    r'\bthings (are|feel) manageable\b', r'\bsleep(ing)? (well|normally)\b',
    r'\bno immediate (concern|issue|danger)\b', r'\bunder control\b',
    r'\b(i am|i\'m) (calm|okay|fine|comfortable|steady)\b', r'\bfine with sleep\b', r'\bfeel safe and\b'
]
DISTRESS_PATTERNS = [
    r'\bworried\b', r'\bstress(ed|ful)?\b', r'\banxious\b', r'\btense\b',
    r"\bcan\'t sleep\b", r'\bcannot sleep\b', r'\btrouble sleeping\b',
    r'\bpanic(king)?\b', r'\boverwhelm(ed|ing)?\b', r'\bafraid\b', r'\bterrified\b',
    r'\bunsafe\b', r'\bthreat(en|ened|ening|s)?\b', r'\bscared\b', r'\btrapped\b',
    r'\bunable to cope\b', r'\bgetting much worse\b'
]
URGENT_PATTERNS = [
    r'\bimmediate danger\b', r'\bneed (urgent|emergency) (help|assistance|protection)\b',
    r'\bthreat(ened|ening)? me (right now|now)\b', r'\bneed protection immediately\b',
    r'\bbeing followed\b', r'\bneed emergency relocation\b', r"\bcan( not|\'t) remain here safely\b"
]

def _hits(patterns, text):
    low = text.lower()
    return [p for p in patterns if re.search(p, low)]

def text_guardrails(text):
    return _hits(POSITIVE_PATTERNS, text), _hits(DISTRESS_PATTERNS, text), _hits(URGENT_PATTERNS, text)


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

def iso_ago(days: float):
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S UTC')

def parse_ts(value):
    try:
        return datetime.strptime(value.replace(' UTC',''), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    except Exception:
        return None

def trajectory(case_id, horizon_days=14):
    with conn() as c:
        rows = c.execute('SELECT created_at, score, risk_label FROM checkins WHERE case_id=? ORDER BY id ASC', (case_id,)).fetchall()
    obs = [(parse_ts(r['created_at']), float(r['score']), r['risk_label']) for r in rows]
    obs = [(t, sc, lab) for t, sc, lab in obs if t is not None]
    if len(obs) < 2:
        return {'projected': False, 'direction': 'insufficient evidence', 'velocity': 0.0,
                'projected_score': None, 'evidence_quality': 'insufficient', 'observations': len(obs)}
    t0, s0, _ = obs[-2]; t1, s1, _ = obs[-1]
    delta_days = max((t1-t0).total_seconds()/86400.0, 1/24)
    velocity = (s1-s0)/delta_days
    projected = max(0.0, min(100.0, s1 + max(-4.0, min(4.0, velocity))*horizon_days))
    direction = 'rising' if velocity > 0.35 else ('falling' if velocity < -0.35 else 'stable')
    return {'projected': True, 'direction': direction, 'velocity': round(velocity,2),
            'projected_score': round(projected,1), 'evidence_quality': 'moderate' if len(obs)<4 else 'strong',
            'observations': len(obs), 'horizon_days': horizon_days}

def intervention_seed(case_id, recommendation, scheme):
    with conn() as c:
        existing = c.execute("SELECT id FROM interventions WHERE case_id=? AND status NOT IN ('completed','dismissed') ORDER BY id DESC LIMIT 1", (case_id,)).fetchone()
        if existing: return existing['id']
        cur = c.execute('INSERT INTO interventions(case_id,created_at,recommendation,scheme,status,assigned_to,completed_at,outcome,notes) VALUES(?,?,?,?,?,?,?,?,?)',
                        (case_id, now(), recommendation, scheme, 'recommended', '', '', '', 'AI recommendation; awaiting authorised human decision.'))
        return cur.lastrowid


def conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

def action_for(label):
    return {
      'stable':'Routine follow-up; continue periodic check-ins.',
      'elevated':'Counsellor review recommended; verify support and next case milestone.',
      'high':'Priority human review; consider counselling, legal-aid/witness-support or safety planning.',
      'critical':'Urgent human review; assess immediate safety and authorised protection/relocation pathways.'
    }[label]

def init_db():
    with conn() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS cases (
          id TEXT PRIMARY KEY, name TEXT, category TEXT, district TEXT, language TEXT,
          baseline REAL DEFAULT 22, case_stage TEXT, safety_status TEXT DEFAULT 'monitored', created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS checkins (
          id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, created_at TEXT, channel TEXT,
          text TEXT, risk_label TEXT, score REAL, model_conf REAL, baseline_delta REAL,
          engagement REAL, latency REAL, voice_rms REAL, voice_zcr REAL, voice_seconds REAL,
          explanation TEXT, recommended_action TEXT, duress INTEGER DEFAULT 0,
          FOREIGN KEY(case_id) REFERENCES cases(id)
        );
        CREATE TABLE IF NOT EXISTS actions (
          id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, created_at TEXT,
          actor TEXT, action TEXT, notes TEXT
        );
        CREATE TABLE IF NOT EXISTS audits (
          id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, actor TEXT, event TEXT, detail TEXT
        );
        CREATE TABLE IF NOT EXISTS interventions (
          id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, created_at TEXT,
          recommendation TEXT, scheme TEXT, status TEXT DEFAULT 'recommended',
          assigned_to TEXT DEFAULT '', completed_at TEXT DEFAULT '', outcome TEXT DEFAULT '', notes TEXT DEFAULT '',
          FOREIGN KEY(case_id) REFERENCES cases(id)
        );
        CREATE TABLE IF NOT EXISTS outreach (
          id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, created_at TEXT, scheduled_at TEXT,
          preferred_channel TEXT, current_channel TEXT, status TEXT DEFAULT 'scheduled',
          attempts INTEGER DEFAULT 0, last_result TEXT DEFAULT '', next_channel TEXT DEFAULT '',
          FOREIGN KEY(case_id) REFERENCES cases(id)
        );
        ''')
        cols = {r['name'] for r in c.execute('PRAGMA table_info(checkins)').fetchall()}
        if 'reviewed_at' not in cols: c.execute('ALTER TABLE checkins ADD COLUMN reviewed_at TEXT')
        if 'reviewed_by' not in cols: c.execute('ALTER TABLE checkins ADD COLUMN reviewed_by TEXT')
        if 'followup_status' not in cols: c.execute("ALTER TABLE checkins ADD COLUMN followup_status TEXT DEFAULT 'Monitoring'")
        case_cols = {r['name'] for r in c.execute('PRAGMA table_info(cases)').fetchall()}
        if 'state' not in case_cols: c.execute("ALTER TABLE cases ADD COLUMN state TEXT DEFAULT 'Maharashtra'")
        if 'preferred_channel' not in case_cols: c.execute("ALTER TABLE cases ADD COLUMN preferred_channel TEXT DEFAULT 'Web'")
        if 'consent_active' not in case_cols: c.execute("ALTER TABLE cases ADD COLUMN consent_active INTEGER DEFAULT 1")
        count = c.execute('SELECT COUNT(*) n FROM cases').fetchone()['n']
        if count == 0:
            seed = [
              ('SAM-1042','Asha K.','Intimidated witness','Mumbai Suburban','Marathi',31,'Pre-testimony','monitored','Maharashtra','SMS'),
              ('SAM-1088','Ravi M.','SC/ST PoA complaint','Pune','Hindi',27,'Investigation','monitored','Maharashtra','IVRS'),
              ('SAM-1120','Meena R.','Atrocity victim','Nashik','Marathi',24,'Rehabilitation','monitored','Maharashtra','App'),
              ('SAM-1147','Demo Critical','Intimidated witness','Thane','English',20,'Pre-testimony','priority','Maharashtra','IVRS'),
              ('SAM-1161','Demo Stable','Family affected by caste violence','Nagpur','Hindi',22,'Investigation','monitored','Maharashtra','SMS'),
              ('SAM-1172','Demo Witness','Threatened witness','Aurangabad','Marathi',29,'Pre-testimony','monitored','Maharashtra','App'),
              ('SAM-1184','Demo Survivor','Grievous hurt survivor','Kolhapur','Marathi',25,'Rehabilitation','monitored','Maharashtra','Web'),
              ('SAM-1195','Demo Complainant','Arson-affected family','Mumbai City','Hindi',26,'Investigation','monitored','Maharashtra','SMS')
            ]
            c.executemany('INSERT INTO cases(id,name,category,district,language,baseline,case_stage,safety_status,state,preferred_channel,consent_active,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                          [(a,b,c1,d,e,f,g,h,i,j,1,iso_ago(60)) for a,b,c1,d,e,f,g,h,i,j in seed])
            hist = [
              ('SAM-1042',35,'I am okay but I am a little tense about the next hearing','elevated',46),
              ('SAM-1042',21,'I am sleeping normally and feel supported','stable',23),
              ('SAM-1042',5,'I feel more worried about testimony and cannot focus well','high',71),
              ('SAM-1088',42,'The delay is stressful but I can manage','elevated',44),
              ('SAM-1088',20,'I feel supported by my family today','stable',21),
              ('SAM-1120',18,'I am anxious about the compensation delay','elevated',45),
              ('SAM-1147',1,'Someone is threatening me right now and I need urgent help','critical',91),
              ('SAM-1161',20,'I feel safe and supported today','stable',22),
              ('SAM-1172',12,'I feel tense about testimony','elevated',47),
              ('SAM-1172',3,'I am increasingly worried and cannot sleep','high',72),
              ('SAM-1184',15,'I am managing but need more support than before','elevated',44),
              ('SAM-1195',10,'I feel safe today and can manage','stable',24)
            ]
            for cid, days_ago, text, label, score in hist:
                c.execute('INSERT INTO checkins(case_id,created_at,channel,text,risk_label,score,model_conf,baseline_delta,engagement,latency,explanation,recommended_action,duress) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                          (cid, iso_ago(days_ago), 'Demo Seed', text, label, score, .92, score-20, 0, 0,
                           'Seeded demonstration record', action_for(label), 1 if label=='critical' else 0))
            c.execute("INSERT INTO interventions(case_id,created_at,recommendation,scheme,status,assigned_to,notes) VALUES(?,?,?,?,?,?,?)",
                      ('SAM-1147',now(),'Urgent human review; assess immediate safety and authorised protection/relocation pathways.','Witness Protection / Safety Planning','recommended','','Seeded demo intervention.'))
            c.execute("INSERT INTO interventions(case_id,created_at,recommendation,scheme,status,assigned_to,notes) VALUES(?,?,?,?,?,?,?)",
                      ('SAM-1172',now(),'Priority human review; consider counselling, legal-aid/witness-support or safety planning.','Counselling / Legal-Aid & Witness Support','approved','Demo Counsellor','Seeded demo intervention.'))
            c.execute('INSERT INTO audits(created_at,actor,event,detail) VALUES(?,?,?,?)', (now(),'system','DEMO_SEED','Seeded synthetic cases with longitudinal history, intervention lifecycle and privacy-safe aggregate demo data.'))
init_db()


def model():
    if not MODEL.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(ROOT/'scripts'/'train_model.py')], check=True)
    return load_model(MODEL)



def voice_stats(path: Path):
    try:
        with wave.open(str(path), 'rb') as w:
            n = w.getnframes(); rate = w.getframerate(); width = w.getsampwidth(); channels = w.getnchannels()
            raw = w.readframes(n)
        if width != 2 or channels != 1 or not raw:
            return {'rms': 0, 'zcr': 0, 'seconds': round(n/max(rate,1),2), 'note':'WAV is not 16-bit mono; limited stats.'}
        samples = struct.unpack('<' + 'h'*(len(raw)//2), raw[:(len(raw)//2)*2])
        rms = (sum(x*x for x in samples) / max(len(samples),1)) ** 0.5
        # zero crossing rate over signed 16-bit samples
        vals = samples
        crossings = sum(1 for a,b in zip(vals, vals[1:]) if (a<0)!=(b<0))
        zcr = crossings/max(len(vals)-1,1)
        return {'rms': round(rms,2), 'zcr': round(zcr,4), 'seconds': round(n/max(rate,1),2), 'note':'Engineering audio statistics only.'}
    except Exception as e:
        return {'rms':0,'zcr':0,'seconds':0,'note':f'Audio read skipped: {e}'}


def analyse(text: str, case_id: str, engagement: float, latency: float, duress: bool, voice: dict):
    m = model()
    probs = m.predict_proba([text])[0]
    classes = list(m.clf.classes_)
    p = {k: float(v) for k, v in zip(classes, probs)}
    model_label = max(p, key=p.get)
    confidence = p[model_label]
    base = RISK_SCORE[model_label]
    positive_hits, distress_hits, urgent_hits = text_guardrails(text)

    if urgent_hits or duress:
        base = max(base, 82 if duress else 78)
    elif positive_hits and not distress_hits:
        base = min(base, 24)
    elif distress_hits:
        base = max(base, 38)
        severe_patterns = [r'\bterrified\b', r'\bunsafe\b', r'\bthreat(en|ened|ening|s)?\b', r"\bcan\'t sleep\b", r'\bcannot sleep\b']
        if any(re.search(pat, text.lower()) for pat in severe_patterns):
            base = max(base, 60)

    with conn() as c:
        row = c.execute('SELECT baseline FROM cases WHERE id=?', (case_id,)).fetchone()
    baseline = float(row['baseline']) if row else 25
    previous = None
    with conn() as c:
        r = c.execute('SELECT score FROM checkins WHERE case_id=? ORDER BY id DESC LIMIT 1', (case_id,)).fetchone()
        if r:
            previous = float(r['score'])
    baseline_delta = (base - baseline) if previous is None else (base - previous) * .55 + (base - baseline) * .45
    engagement_delta = max(0, min(12, (engagement - 55) * .16))
    latency_delta = max(0, min(10, (latency - 30) * .12))
    voice_delta = 0
    if voice.get('rms', 0) > 7000: voice_delta += 3
    if voice.get('zcr', 0) > .16: voice_delta += 2
    if duress: voice_delta += 18
    raw = base + baseline_delta * .22 + engagement_delta + latency_delta + voice_delta
    score = max(0, min(100, round(raw, 1)))
    if urgent_hits or duress or score >= 82:
        final = 'critical'
    elif score >= 62:
        final = 'high'
    elif score >= 35 or distress_hits:
        final = 'elevated'
    else:
        final = 'stable'
    if distress_hits and not urgent_hits and not duress and score < 35:
        score = 35.0
        if final == 'stable':
            final = 'elevated'
    if positive_hits and not distress_hits and not urgent_hits and not duress:
        final = 'stable'; score = min(score, 30)
    terms = []
    try:
        terms = m.top_terms(text, model_label, limit=5)
    except Exception:
        pass
    explanation = json.dumps({
        'model_top_terms': terms,
        'model_probabilities': {k: round(v, 3) for k, v in p.items()},
        'baseline_delta': round(baseline_delta, 1),
        'engagement_delta': round(engagement_delta, 1),
        'latency_delta': round(latency_delta, 1),
        'voice_delta': voice_delta,
        'duress_signal': duress,
        'guardrail_positive_signals': len(positive_hits),
        'guardrail_distress_signals': len(distress_hits),
        'guardrail_urgent_signals': len(urgent_hits),
        'guardrail_note': 'Transparent lexical sanity layer for engineering demo; not clinical validation.'
    })
    return final, score, confidence, baseline_delta, explanation, action_for(final), p


def audit(actor,event,detail):
    with conn() as c: c.execute('INSERT INTO audits(created_at,actor,event,detail) VALUES(?,?,?,?)',(now(),actor,event,detail))


def base_context(request, **extra):
    return {'request':request,'languages':LANGUAGES, **extra}

def dashboard_data():
    with conn() as c:
        cases = c.execute('''SELECT ca.*, COALESCE((SELECT score FROM checkins ch WHERE ch.case_id=ca.id ORDER BY ch.id DESC LIMIT 1), ca.baseline) score,
          COALESCE((SELECT risk_label FROM checkins ch WHERE ch.case_id=ca.id ORDER BY ch.id DESC LIMIT 1),'stable') risk_label FROM cases ca ORDER BY score DESC''').fetchall()
        stats = { 'cases': c.execute('SELECT COUNT(*) n FROM cases').fetchone()['n'], 'checkins':c.execute('SELECT COUNT(*) n FROM checkins').fetchone()['n'], 'priority':c.execute("SELECT COUNT(*) n FROM checkins WHERE risk_label IN ('high','critical') AND id IN (SELECT MAX(id) FROM checkins GROUP BY case_id)").fetchone()['n'], 'actions':c.execute('SELECT COUNT(*) n FROM actions').fetchone()['n'] }
    return cases, stats

@app.get('/', response_class=HTMLResponse)
def dashboard(request: Request):
    cases, stats = dashboard_data()
    return templates.TemplateResponse(request, 'dashboard.html', base_context(request,cases=cases,stats=stats,role=''))

@app.get('/victim', response_class=HTMLResponse)
def victim_portal(request: Request):
    with conn() as c: cases=c.execute('SELECT id, language FROM cases ORDER BY id').fetchall()
    return templates.TemplateResponse(request, 'checkin.html', base_context(request,cases=cases,role='victim',active='victim'))

@app.get('/victim/help', response_class=HTMLResponse)
def victim_help(request: Request):
    return templates.TemplateResponse(request, 'help.html', base_context(request,role='victim',active='victim'))

@app.get('/victim/history', response_class=HTMLResponse)
def victim_history_default(request: Request, case_id: Optional[str] = None):
    # Demo-only selector: production would derive case_id from the authenticated victim identity.
    with conn() as c:
        selected = case_id or c.execute('SELECT id FROM cases ORDER BY id LIMIT 1').fetchone()['id']
        case=c.execute('SELECT id,name,case_stage,safety_status FROM cases WHERE id=?',(selected,)).fetchone()
        checks=c.execute('''SELECT id,created_at,channel,text,risk_label,reviewed_at,followup_status
                           FROM checkins WHERE case_id=? ORDER BY id DESC''',(selected,)).fetchall()
        all_cases=c.execute('SELECT id FROM cases ORDER BY id').fetchall()
    if not case: return RedirectResponse('/victim', status_code=303)
    return templates.TemplateResponse(request, 'victim_history.html', base_context(request, role='victim', active='history', case=case, checks=checks, all_cases=all_cases))

@app.get('/victim/history/{case_id}', response_class=HTMLResponse)
def victim_history(request: Request, case_id: str):
    # Demo-only case-scoped history. A production deployment would resolve case_id
    # from the authenticated victim identity/session rather than a URL parameter.
    with conn() as c:
        case=c.execute('SELECT id,name,case_stage,safety_status FROM cases WHERE id=?',(case_id,)).fetchone()
        checks=c.execute('''SELECT id,created_at,channel,text,risk_label,reviewed_at,followup_status
                           FROM checkins WHERE case_id=? ORDER BY id DESC''',(case_id,)).fetchall()
    if not case: return RedirectResponse('/victim', status_code=303)
    return templates.TemplateResponse(request, 'victim_history.html', base_context(request, role='victim', active='victim', case=case, checks=checks))

@app.get('/counsellor', response_class=HTMLResponse)
def counsellor_console(request: Request):
    cases, stats = dashboard_data()
    return templates.TemplateResponse(request, 'counsellor.html', base_context(request,cases=cases,stats=stats,role='counsellor',active='counsellor'))

@app.get('/checkin', response_class=HTMLResponse)
def checkin_page(request: Request):
    with conn() as c: cases=c.execute('SELECT * FROM cases ORDER BY id').fetchall()
    return templates.TemplateResponse(request, 'checkin.html', base_context(request,cases=cases,role='victim',active='victim'))

@app.post('/checkin')
async def checkin(case_id: str=Form(...), text: str=Form(...), engagement: float=Form(55), latency: float=Form(20), language: str=Form('English'), channel: str=Form('Web/Chat'), duress: bool=Form(False), audio: Optional[UploadFile]=File(None)):
    voice={'rms':0,'zcr':0,'seconds':0,'note':'No audio supplied.'}
    if audio and audio.filename:
        p=UPLOADS/(uuid.uuid4().hex+'.wav'); p.write_bytes(await audio.read()); voice=voice_stats(p)
    label,score,conf,bd,explanation,action,probs=analyse(text,case_id,engagement,latency,duress,voice)
    with conn() as c:
        cur = c.execute('''INSERT INTO checkins(case_id,created_at,channel,text,risk_label,score,model_conf,baseline_delta,engagement,latency,voice_rms,voice_zcr,voice_seconds,explanation,recommended_action,duress) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
          (case_id,now(),f'{channel} · {language}',text,label,score,conf,bd,engagement,latency,voice['rms'],voice['zcr'],voice['seconds'],explanation,action,int(duress)))
        checkin_id = cur.lastrowid
    if label in ('high','critical'):
        scheme = 'Witness Protection / Safety Planning' if label == 'critical' else 'Counselling / Legal-Aid & Witness Support'
        intervention_seed(case_id, action, scheme)
    audit('AI-engine','CHECKIN_ANALYSED',f'{case_id}: check-in {checkin_id}: {label} / {score} / confidence {conf:.2f}')
    return RedirectResponse(f'/victim/received/{case_id}/{checkin_id}', status_code=303)

@app.get('/victim/received/{case_id}', response_class=HTMLResponse)
def victim_received_legacy(request: Request, case_id: str):
    with conn() as c:
        latest=c.execute('SELECT * FROM checkins WHERE case_id=? ORDER BY id DESC LIMIT 1',(case_id,)).fetchone()
    if not latest: return RedirectResponse('/victim', status_code=303)
    return RedirectResponse(f'/victim/received/{case_id}/{latest["id"]}', status_code=307)

@app.get('/victim/received/{case_id}/{checkin_id}', response_class=HTMLResponse)
def victim_received(request: Request, case_id: str, checkin_id: int):
    with conn() as c:
        latest=c.execute('SELECT * FROM checkins WHERE id=? AND case_id=?',(checkin_id,case_id)).fetchone()
        case=c.execute('SELECT id,name FROM cases WHERE id=?',(case_id,)).fetchone()
    if not latest or not case: return RedirectResponse('/victim', status_code=303)
    return templates.TemplateResponse(request, 'received.html', base_context(request,role='victim',active='victim',case_id=case_id,checkin_id=checkin_id,case=case,latest=latest))

@app.get('/case/{case_id}', response_class=HTMLResponse)
def case_page(request: Request, case_id: str):
    with conn() as c:
        case=c.execute('SELECT * FROM cases WHERE id=?',(case_id,)).fetchone()
        checks=c.execute('SELECT * FROM checkins WHERE case_id=? ORDER BY id DESC',(case_id,)).fetchall()
        acts=c.execute('SELECT * FROM actions WHERE case_id=? ORDER BY id DESC',(case_id,)).fetchall()
    if not case: return HTMLResponse('Case not found',404)
    latest=checks[0] if checks else None
    expl={}
    if latest and latest['explanation']:
        try:
            expl=json.loads(latest['explanation'])
        except (json.JSONDecodeError, TypeError):
            # Older/demo seed records may contain plain-text explanations.
            expl={'legacy_note': latest['explanation']}
    traj = trajectory(case_id)
    with conn() as c:
        interventions = c.execute('SELECT * FROM interventions WHERE case_id=? ORDER BY id DESC',(case_id,)).fetchall()
        outreach_rows = c.execute('SELECT * FROM outreach WHERE case_id=? ORDER BY id DESC',(case_id,)).fetchall()
    return templates.TemplateResponse(request, 'case.html', base_context(request,case=case,checks=checks,acts=acts,latest=latest,expl=expl,traj=traj,interventions=interventions,outreach=outreach_rows,role='counsellor',active='queue'))

@app.post('/case/{case_id}/action')
def case_action(case_id: str, action: str=Form(...), notes: str=Form(''), reviewed: bool=Form(False), followup_status: str=Form('Monitoring')):
    with conn() as c:
        latest = c.execute('SELECT id FROM checkins WHERE case_id=? ORDER BY id DESC LIMIT 1', (case_id,)).fetchone()
        c.execute('INSERT INTO actions(case_id,created_at,actor,action,notes) VALUES(?,?,?,?,?)', (case_id, now(), 'Human Reviewer', action, notes))
        if latest and reviewed:
            c.execute('UPDATE checkins SET reviewed_at=?, reviewed_by=?, followup_status=? WHERE id=?', (now(), 'Human Reviewer', followup_status, latest['id']))
    lifecycle = 'approved' if reviewed else 'recommended'
    scheme = 'Witness Protection / Safety Planning' if ('protection' in action.lower() or 'safety' in action.lower()) else ('Legal Aid / Witness Support' if 'legal' in action.lower() else 'Counselling / Welfare Support')
    with conn() as c:
        active_int = c.execute("SELECT id FROM interventions WHERE case_id=? AND status NOT IN ('completed','dismissed') ORDER BY id DESC LIMIT 1",(case_id,)).fetchone()
        if active_int:
            c.execute("UPDATE interventions SET status=?, assigned_to=?, notes=? WHERE id=?",
                      (lifecycle,'Human Reviewer',notes,active_int['id']))
        else:
            c.execute('INSERT INTO interventions(case_id,created_at,recommendation,scheme,status,assigned_to,completed_at,outcome,notes) VALUES(?,?,?,?,?,?,?,?,?)',
                      (case_id,now(),action,scheme,lifecycle,'Human Reviewer','', '',notes))
    audit('Human Reviewer','ACTION_RECORDED',f'{case_id}: {action}; intervention lifecycle={lifecycle}')
    if reviewed:
        audit('Human Reviewer','CHECKIN_REVIEWED',f'{case_id}: latest check-in reviewed; victim status updated.')
    return RedirectResponse(f'/case/{case_id}', status_code=303)

@app.get('/queue', response_class=HTMLResponse)
def queue(request: Request):
    with conn() as c:
        rows=c.execute('''SELECT ca.*, ch.score, ch.risk_label, ch.created_at checkin_at, ch.recommended_action FROM cases ca JOIN checkins ch ON ch.id=(SELECT MAX(id) FROM checkins WHERE case_id=ca.id) ORDER BY ch.score DESC''').fetchall()
    return templates.TemplateResponse(request, 'queue.html',base_context(request,rows=rows,role='counsellor',active='queue'))

@app.post('/case/{case_id}/intervention/{intervention_id}')
def intervention_action(case_id: str, intervention_id: int, status: str=Form(...), outcome: str=Form(''), notes: str=Form('')):
    allowed = {'recommended','approved','dispatched','completed','dismissed'}
    if status not in allowed:
        return RedirectResponse(f'/case/{case_id}', status_code=303)
    with conn() as c:
        c.execute('UPDATE interventions SET status=?, outcome=?, notes=?, assigned_to=?, completed_at=? WHERE id=? AND case_id=?',
                  (status,outcome,notes,'Human Reviewer',now() if status=='completed' else '',intervention_id,case_id))
    audit('Human Reviewer','INTERVENTION_LIFECYCLE',f'{case_id}: intervention {intervention_id} -> {status}')
    return RedirectResponse(f'/case/{case_id}', status_code=303)

@app.post('/case/{case_id}/outreach')
def schedule_outreach(case_id: str, channel: str=Form('SMS')):
    channels = ['Web','App','SMS','IVRS']
    if channel not in channels: channel='SMS'
    with conn() as c:
        case = c.execute('SELECT preferred_channel,consent_active FROM cases WHERE id=?',(case_id,)).fetchone()
        if not case: return RedirectResponse('/queue',status_code=303)
        if not case['consent_active']:
            audit('System','OUTREACH_BLOCKED',f'{case_id}: consent inactive')
            return RedirectResponse(f'/case/{case_id}',status_code=303)
        order = channels[channels.index(channel):] + channels[:channels.index(channel)]
        next_channel = order[1] if len(order)>1 else ''
        c.execute('INSERT INTO outreach(case_id,created_at,scheduled_at,preferred_channel,current_channel,status,attempts,last_result,next_channel) VALUES(?,?,?,?,?,?,?,?,?)',
                  (case_id,now(),iso_ago(-1),channel,channel,'scheduled',0,'',next_channel))
    audit('Human Reviewer','OUTREACH_SCHEDULED',f'{case_id}: preferred={channel}; fallback={next_channel}')
    return RedirectResponse(f'/case/{case_id}',status_code=303)

@app.post('/outreach/{outreach_id}/simulate')
def simulate_outreach(outreach_id: int, result: str=Form('delivered')):
    with conn() as c:
        row=c.execute('SELECT * FROM outreach WHERE id=?',(outreach_id,)).fetchone()
        if not row: return RedirectResponse('/queue',status_code=303)
        if result=='delivered':
            c.execute("UPDATE outreach SET status='delivered', attempts=attempts+1, last_result='Delivered' WHERE id=?",(outreach_id,))
        elif result=='responded':
            c.execute("UPDATE outreach SET status='responded', attempts=attempts+1, last_result='Victim responded' WHERE id=?",(outreach_id,))
        else:
            nxt=row['next_channel']
            c.execute("UPDATE outreach SET status=?, attempts=attempts+1, last_result=?, current_channel=?, next_channel=? WHERE id=?",
                      ('fallback_pending' if nxt else 'counsellor_flag','Channel failed',nxt or row['current_channel'],'',outreach_id))
            if not nxt:
                audit('System','OUTREACH_EXHAUSTED',f"{row['case_id']}: all configured channels exhausted; continuity review required.")
                c.execute("INSERT INTO actions(case_id,created_at,actor,action,notes) VALUES(?,?,?,?,?)",
                          (row['case_id'],now(),'System','Contact continuity review','All configured outreach channels exhausted.'))
    audit('System','OUTREACH_RESULT',f"{row['case_id']}: outreach {outreach_id} -> {result}")
    return RedirectResponse(f"/case/{row['case_id']}",status_code=303)

@app.get('/admin', response_class=HTMLResponse)
def admin_dashboard(request: Request):
    with conn() as c:
        total=c.execute('SELECT COUNT(*) n FROM cases').fetchone()['n']
        checkins=c.execute('SELECT COUNT(*) n FROM checkins').fetchone()['n']
        interventions=c.execute("SELECT COUNT(*) n FROM interventions").fetchone()['n']
        completed=c.execute("SELECT COUNT(*) n FROM interventions WHERE status='completed'").fetchone()['n']
        rows=c.execute("SELECT state, district, COUNT(*) n, SUM(CASE WHEN ch.risk_label IN ('high','critical') THEN 1 ELSE 0 END) high_count FROM cases ca LEFT JOIN checkins ch ON ch.id=(SELECT MAX(id) FROM checkins WHERE case_id=ca.id) GROUP BY state,district ORDER BY state,district").fetchall()
    def cell(n): return '<5' if n < 5 else str(n)
    geo=[{'state':r['state'],'district':r['district'],'cases':cell(r['n']),'priority':cell(r['high_count'])} for r in rows]
    return templates.TemplateResponse(request,'admin.html',base_context(request,role='admin',active='admin',
        summary={'cases':cell(total),'checkins':cell(checkins),'interventions':cell(interventions),'completed':cell(completed)},geo=geo))

@app.get('/validation', response_class=HTMLResponse)
def validation(request: Request):
    metrics=json.loads(METRICS.read_text()) if METRICS.exists() else {}
    return templates.TemplateResponse(request, 'validation.html',base_context(request,m=metrics,role='counsellor'))

@app.get('/audit', response_class=HTMLResponse)
def audit_page(request: Request):
    with conn() as c: rows=c.execute('SELECT * FROM audits ORDER BY id DESC LIMIT 80').fetchall()
    return templates.TemplateResponse(request, 'audit.html',base_context(request,rows=rows,role='counsellor'))

@app.get('/integrations', response_class=HTMLResponse)
def integrations(request: Request):
    adapters=[
      ('NHAA 14566','Inbound case/victim reference + approved communication channel','Mock adapter','Ready for authorised API contract'),
      ('e-Courts','Case stage / hearing / testimony milestone context','Mock adapter','Integration point defined; no live government data'),
      ('PFMS','Relief/compensation status signal for follow-up','Mock adapter','Read-only demo status; production access requires authorisation'),
      ('Bhashini','Multilingual speech/text routing','Adapter interface','Demo language routing; production API credentials required')]
    return templates.TemplateResponse(request, 'integrations.html',base_context(request,adapters=adapters,role='counsellor'))

@app.get('/api/health')
def health():
    return {'status':'ok','service':'Samarthan','version':'0.8.0',
            'model':'TF-IDF + Logistic Regression','environment':'engineering POC',
            'features':['dynamic distress triage','early-warning trajectory','closed-loop interventions','consent-gated channel fallback','aggregate admin monitoring']}

@app.get('/api/victim-status/{case_id}/{checkin_id}')
def victim_status(case_id: str, checkin_id: int):
    with conn() as c:
        row = c.execute('''SELECT ch.id,ch.created_at,ch.reviewed_at,ch.reviewed_by,ch.followup_status,ca.case_stage
                           FROM checkins ch JOIN cases ca ON ca.id=ch.case_id
                           WHERE ch.id=? AND ch.case_id=?''', (checkin_id, case_id)).fetchone()
    if not row:
        return JSONResponse({'found': False}, status_code=404)
    return {'found': True, 'checkin_id': row['id'], 'reviewed': bool(row['reviewed_at']), 'reviewed_at': row['reviewed_at'],
            'followup_status': row['followup_status'] or 'Monitoring', 'case_stage': row['case_stage']}

@app.get('/api/cases')
def api_cases():
    with conn() as c: rows=c.execute('SELECT * FROM cases').fetchall()
    return [dict(r) for r in rows]

@app.post('/api/duress')
def api_duress(case_id: str, signal: str='#'):
    if signal != '#': return JSONResponse({'accepted':False,'reason':'Demo silent-duress trigger is #'},status_code=400)
    audit('Channel Adapter','SILENT_DURESS',f'{case_id}: DTMF # received; priority human review required.')
    return {'accepted':True,'case_id':case_id,'signal':'#','routing':'priority_human_review'}
