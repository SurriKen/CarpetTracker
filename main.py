import asyncio
import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from fastapi.background import BackgroundTasks
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from predict_sync_videos import predict
from predict import video_paths
import parameters

app = FastAPI()

app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')


# datetime processing
def get_time(tm) -> str:
    ret = ''
    if tm is not None:
        try:
            ret = tm.strftime('%H:%M:%S, ') + str(tm.day) + ' ' +\
                  list(['янв', 'фев', 'мар' 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек'])[tm.month]
        except:
            pass
    return ret


async def background_service():
    """
    Background service
    """

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, predict, video_paths, False, True)
        parameters.started = False


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    """
    Main webpage
    """

    print('host:', request.client.host)
    service_status = 'запущен' if parameters.started else 'остановлен'
    service_class = 'text-success' if parameters.started else 'text-danger'
    btn_label = 'Стоп' if parameters.started else 'Старт'
    btn_class = 'btn-danger' if parameters.started else 'btn-success'

    context = {'request': request,
               'service_status': service_status, 'service_class': service_class,
               'btn_label': btn_label, 'btn_class': btn_class,
               'start_time': get_time(parameters.start_time),
               'stop_time': get_time(parameters.stop_time),
               'latest_results': parameters.latest_results}

    return templates.TemplateResponse('index.html', context=context)


@app.post('/start/')
async def start_service(background_tasks: BackgroundTasks):
    """
    Service start
    """

    if not parameters.started:
        parameters.started = True
        parameters.start_time = datetime.datetime.now()
        parameters.stop_time = None
        parameters.latest_results = {key: 0 for key, value in parameters.latest_results.items()}
        background_tasks.add_task(background_service)  # starting a background task
        return {'status': 200, 'message': 'Service started successfully'}
    else:
        return {'status': 200, 'message': 'Service is already running'}


@app.post('/stop/')
async def stop_service():
    """
    Service stop
    """

    if parameters.started:
        parameters.started = False  # task start flag
        parameters.stop_time = datetime.datetime.now()
        return {'status': 200, 'message': 'Service stopped successfully'}
    else:
        return {'status': 200, 'message': 'Service is not running'}


@app.get('/status/')
async def get_service_status():
    """
    Get service status and latest results
    """

    dict1 = {'status': 200, 'started': parameters.started,
             'start_time': get_time(parameters.start_time),
             'stop_time': get_time(parameters.stop_time)}

    dict2 = {f'count{key}' if key != 'total_count' else key: value for key, value in parameters.latest_results.items()}
    dict1.update(dict2)

    return dict1
