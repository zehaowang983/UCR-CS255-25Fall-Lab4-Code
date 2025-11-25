from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from starlette.middleware.sessions import SessionMiddleware

from contextlib import asynccontextmanager

from google_auth_oauthlib.flow import Flow

from sqlalchemy import create_engine, Column, String, Float, LargeBinary, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

import secrets
import torch
import math
import sqlite3
import logging
import os
import io

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    email = Column(String, primary_key=True, nullable=False)
    name = Column(String, nullable=False)

class Submission(Base):
    __tablename__ = 'submissions'
    id = Column(String, primary_key=True, default=lambda: secrets.token_urlsafe(32), nullable=False)
    alg = Column(String, nullable=False)
    status = Column(String, nullable=False)
    file = Column(LargeBinary, nullable=False)
    timestamp = Column(String, nullable=False)

    tp1_diff = Column(Float)
    tp5_diff = Column(Float)
    eps = Column(Float)
    score = Column(Float)
    tp1_raw_score = Column(Float)
    tp5_raw_score = Column(Float)
    finish_timestamp = Column(String)
    comment = Column(String)
    traceback = Column(String)

    user_email = Column(String, ForeignKey('users.email'))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, timezone, device, model, testset, templates, func_ptrs, submit_ddl, session
    global CLIENT_SECRET, SCOPES, MAX_SUBMISSIONS, TEMPLATES_DIR, RESNET_MODEL, TIME_ZONE, TESTSET, SUBMISSION_DIR, SUBMIT_DDL

    logger = logging.getLogger('uvicorn.error')

    load_dotenv(override=True)

    DB_FILE = os.getenv('DB_FILE')
    CLIENT_SECRET = os.getenv('CLIENT_SECRET')
    SCOPES = os.getenv('SCOPES').split(',')
    MAX_FGSM_SUBMISSIONS = int(os.getenv('MAX_FGSM_SUBMISSIONS'))
    MAX_PGD_SUBMISSIONS = int(os.getenv('MAX_PGD_SUBMISSIONS'))
    MAX_SUBMISSIONS = {'fgsm': MAX_FGSM_SUBMISSIONS, 'pgd': MAX_PGD_SUBMISSIONS}
    TEMPLATES_DIR = os.getenv('TEMPLATES_DIR')
    RESNET_MODEL = os.getenv('RESNET_MODEL')
    TIME_ZONE = os.getenv('TIME_ZONE')
    TESTSET = os.getenv('TESTSET')
    # SUBMISSION_DIR = os.getenv('SUBMISSION_DIR')
    SUBMIT_DDL = os.getenv('SUBMIT_DDL')

    logger.info(f'Using config: \n \
            DB_FILE: {DB_FILE}\n \
            CLIENT_SECRET: {CLIENT_SECRET}\n \
            SCOPES: {SCOPES}\n \
            MAX_FGSM_SUBMISSIONS: {MAX_FGSM_SUBMISSIONS}\n \
            MAX_PGD_SUBMISSIONS: {MAX_PGD_SUBMISSIONS}\n \
            TEMPLATES_DIR: {TEMPLATES_DIR}\n \
            RESNET_MODEL: {RESNET_MODEL}\n \
            TIME_ZONE: {TIME_ZONE}\n \
            TESTSET: {TESTSET}\n \
            SUBMIT_DDL: {SUBMIT_DDL}\n')
    
    submit_ddl = datetime.fromisoformat(SUBMIT_DDL)

    engine = create_engine(f'sqlite:///{DB_FILE}')  
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    timezone = ZoneInfo(TIME_ZONE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(RESNET_MODEL, map_location=device)
    model.eval()
    testset = torch.load(TESTSET, weights_only=True)

    templates = Jinja2Templates(directory=TEMPLATES_DIR)

    yield

    session.close()
    engine.dispose()

def get_total_grade(user_email):
    submissions = session.query(Submission).filter(
        Submission.user_email == user_email,
        Submission.status == 'Complete'
    ).all()
    if not submissions:
        return 0
    fgsm_grade = 0
    pgd_grade = 0
    for submission in submissions:
        if submission.alg == 'fgsm':
            fgsm_grade = max(fgsm_grade, submission.score)
        elif submission.alg == 'pgd':
            pgd_grade = max(pgd_grade, submission.score)
    return fgsm_grade + pgd_grade

def get_submissions_left(user_email, alg):
    submissions = session.query(Submission).filter(
        Submission.user_email == user_email,
        Submission.alg == alg,
        Submission.status != 'Error'
    ).all()
    return MAX_SUBMISSIONS[alg] - len(submissions)

def evaluate_data(adv_imgs, batch_size=32, workers=8):
    if len(adv_imgs) != len(testset):
        raise ValueError("The number of adversarial images should be the same as the number of test images.")
    zipped = [(a.detach(), b, c) for a, (b, c) in zip(adv_imgs, testset)]
    zipped_data = torch.utils.data.DataLoader(zipped, batch_size=batch_size, shuffle=False, num_workers=workers)

    original_tp, original_tp5, adv_tp, adv_tp5 = .0, .0, .0, .0
    eps = .0
    for adv, ori, labels in zipped_data:
        adv, ori, labels = adv.to('cuda:0'), ori.to('cuda:0'), labels.to('cuda:0')
        adv_output, ori_output = model(adv), model(ori)
        adv_tp += (adv_output.argmax(dim=1) == labels).sum().item()
        adv_tp5 += (adv_output.topk(5, dim=1)[1] == labels.view(-1, 1)).sum().item()
        original_tp += (ori_output.argmax(dim=1) == labels).sum().item()
        original_tp5 += (ori_output.topk(5, dim=1)[1] == labels.view(-1, 1)).sum().item()
        eps += torch.abs(adv - ori).sum().item()
    eps /= len(adv_imgs)*adv_imgs[0].numel()
    
    adv_acc = adv_tp / len(adv_imgs)
    adv_acc5 = adv_tp5 / len(adv_imgs)
    original_acc = original_tp / len(adv_imgs)
    original_acc5 = original_tp5 / len(adv_imgs)
    return original_acc-adv_acc, original_acc5-adv_acc5, eps

def evaluate(submission_id):
    try:
        submission = session.query(Submission).filter(Submission.id == submission_id).first()
        submission_file = submission.file
        file_like_obj = io.BytesIO(submission_file)
        adv_imgs = torch.load(file_like_obj, weights_only=True, map_location='cpu')
        diff_tp, diff_tp5, eps = evaluate_data(adv_imgs)
        tp1_score = 0.33 + 33*diff_tp-math.exp(33*eps) if submission.alg == 'fgsm' else 26*diff_tp-math.exp(66*eps)
        tp5_score = 66*diff_tp5-math.exp(33*eps) if submission.alg == 'fgsm' else 46*diff_tp5-math.exp(66*eps)

        submission.status = 'Complete'
        submission.tp1_diff = diff_tp
        submission.tp5_diff = diff_tp5
        submission.eps = eps
        submission.score = min(max(0, max(tp1_score, tp5_score)), 10) if submission.alg == 'fgsm' else 0.2*min(max(0, min(tp1_score, tp5_score)), 10)
        submission.tp1_raw_score = tp1_score
        submission.tp5_raw_score = tp5_score
    except Exception as e:
        submission.status = 'Error'
        submission.comment = 'Failed to evaluate submission. Is it persisted by torch.save() and in correct format, i.e., List[torch.tensor[3*32*32]]?'
        submission.traceback = str(e)

    submission.finish_timestamp = datetime.now(timezone).isoformat()
    session.commit()

secret_key = secrets.token_urlsafe(32)
app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=secret_key, max_age=1800)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        'submit.html', 
        {
            'request': request,
        }
    )

@app.get("/auth")
async def auth(request: Request):
    user_email = request.session.get('user_email')
    if user_email:
        return RedirectResponse(request.url_for('root'))

    redirect_uri = request.url_for('auth_callback')

    flow = Flow.from_client_secrets_file(
        CLIENT_SECRET,
        scopes=SCOPES,
        redirect_uri=redirect_uri
    )
    authorization_url, state = flow.authorization_url()
    request.session['state'] = state
    return RedirectResponse(authorization_url)

@app.get("/auth/callback")
async def auth_callback(request: Request):
    try:
        state = request.query_params['state']
        if state != request.session['state']:
            raise HTTPException(status_code=400, detail='Invalid state')

        code = request.query_params['code']
        if not code:
            raise HTTPException(status_code=400, detail='Login Failed')
        redirect_uri = request.url_for('auth_callback')
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRET,
            scopes=SCOPES,
            redirect_uri=redirect_uri
        )
        flow.fetch_token(code=code)
        credentials = flow.credentials

        from google.oauth2.id_token import verify_oauth2_token
        from google.auth.transport.requests import Request as GoogleRequest

        user_info = verify_oauth2_token(
            credentials.id_token, GoogleRequest(), flow.client_config['client_id']
        )
        email = user_info.get('email')
        name = user_info.get('name')

        user = session.query(User).filter(User.email == email).all()
        if not user:
            user = User(email=email, name=name)
            session.add(user)
            session.commit()
        elif len(user) > 1:
            raise HTTPException(status_code=500, detail='Multiple users with same email found, contact TA :(.')
        else:
            user = user[0]
        
        request.session['user_email'] = user.email
        return RedirectResponse(request.url_for('auth'))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(request.url_for('root'))

@app.post("/submit")
async def submit(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    alg: str = Form(...),
):
    submit_ts = datetime.now(timezone)
    if submit_ts > submit_ddl:
        raise HTTPException(status_code=403, detail='Submission Deadline Passed')

    user_email = request.session.get('user_email')
    if not user_email:
        raise HTTPException(status_code=401, detail='Unauthorized')

    if get_submissions_left(user_email, alg) <= 0:
        raise HTTPException(status_code=429, detail='Submission Chance Limit Exceeded')
    
    content = []
    file_size = 0
    chunk_size = 1024 * 1024

    submission = Submission(
        user_email=user_email,
        alg=alg,
        status='Submitted',
        timestamp=submit_ts.isoformat()
    )

    while chunk := await file.read(chunk_size):
        file_size += len(chunk)
        if file_size > 3 * 1024 * 1024:
            submission.status = 'Error'
            submission.comment = 'File size exceeded 3MB'
            session.add(submission)
            session.commit()
            raise HTTPException(status_code=413, detail='File size Exceeded 3MB')
        content.append(chunk)
    
    submission.file = b''.join(content)
    submission.status = 'Submitted'
    session.add(submission)
    session.commit()

    background_tasks.add_task(evaluate, submission.id)
    return JSONResponse({'submission_id': submission.id}, status_code=202)

@app.get("/submissions")
async def get_submissions(request: Request):
    user_email = request.session.get('user_email')
    if not user_email:
        return JSONResponse(None)
    submissions = session.query(Submission).filter(Submission.user_email == user_email).all()
    return JSONResponse([{
        'id': submission.id,
        'alg': submission.alg,
        'status': submission.status,
        'timestamp': submission.timestamp,
        'score': submission.score
    } for submission in submissions])

@app.get("/submission/{submission_id}")
async def get_submission_detail(request: Request, submission_id: str):
    user_email = request.session.get('user_email')
    if not user_email:
        raise HTTPException(status_code=401, detail='Unauthorized')
    submission = session.query(Submission).filter(
        Submission.id == submission_id,
        Submission.user_email == user_email
    ).first()
    if not submission:
        raise HTTPException(status_code=404, detail='Submission Not Found')
    return JSONResponse({
        'tp1_diff': submission.tp1_diff,
        'tp5_diff': submission.tp5_diff,
        'eps': submission.eps,
        'tp1_raw_score': submission.tp1_raw_score,
        'tp5_raw_score': submission.tp5_raw_score,
        'finish_timestamp': submission.finish_timestamp,
        'comment': submission.comment
    })

@app.get("/user")
async def get_user(request: Request):
    user_email = request.session.get('user_email')
    if not user_email:
        return JSONResponse(None)

    user = session.query(User).filter(User.email == user_email).first()
    return JSONResponse({
        'email': user.email,
        'name': user.name,
        'fgsm_submissions_left': get_submissions_left(user_email, 'fgsm'),
        'pgd_submissions_left': get_submissions_left(user_email, 'pgd'),
        'submit_ddl': submit_ddl.isoformat(),
        'total_grade': get_total_grade(user_email)
    })