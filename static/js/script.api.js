$(document).ready(function() {
    updateLabels();

    $('#toggleBtn').click(function() {
        var action = $('#toggleBtn').html() === 'Старт' ? 'start' : 'stop';
        $.post('/' + action + '/', function() {
            updateLabels();
        });
    });

    setInterval(function() {
        updateLabels();
    }, 1000);

    function updateLabels() {
        $.get('/status/')
         .done(function(response) {
            if (response.started) {
                $('#statusSrv').html('запущен');
                $('#statusSrv').removeClass('text-danger').addClass('text-success');
                $('#toggleBtn').html('Стоп');
                $('#toggleBtn').removeClass('btn-success').addClass('btn-danger');
            } else {
                $('#statusSrv').html('остановлен');
                $('#statusSrv').removeClass('text-success').addClass('text-danger');
                $('#toggleBtn').html('Старт');
                $('#toggleBtn').removeClass('btn-danger').addClass('btn-success');
            }
            $('#start_time').html(response.start_time);
            $('#stop_time').html(response.stop_time);
            $('#count60x90').html(response.count60x90);
            $('#count85x150').html(response.count85x150);
            $('#count115x200').html(response.count115x200);
            $('#count115x400').html(response.count115x400);
            $('#count150x300').html(response.count150x300);
            $('#total_count').html(response.total_count);
        })
        .fail(function(jqXHR, textStatus, errorThrown) {
            $('#statusSrv').html('сервер не отвечает');
            $('#statusSrv').removeClass('text-success').addClass('text-danger');
            $('#toggleBtn').html('Старт');
            $('#toggleBtn').removeClass('btn-danger').addClass('btn-success');
        });
    }
});
