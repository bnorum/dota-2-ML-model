// main.js
$(document).ready(function() {
    $('#analyze-form').on('submit', function(event) {
        event.preventDefault();
        var match_id = $('input[name="match_id"]').val();
        $.post('/analyze', {match_id: match_id}, function(data) {
            $('#result').html('Radiant Win Prediction: ' + data.radiant_win);
        });
    });
});
