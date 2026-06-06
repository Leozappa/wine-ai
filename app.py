from flask import Flask, request, render_template_string, send_file
import os
import sys

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads_temp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>Wine AI</title>
<style>
body { font-family: Georgia, serif; background: #F8F3EC; color: #3D0A14; }
header { background: #3D0A14; color: #C9A84C; text-align: center; padding: 30px; }
.container { max-width: 700px; margin: 40px auto; padding: 0 20px; }
.card { background: white; border-radius: 12px; padding: 35px; text-align: center; box-shadow: 0 4px 20px rgba(61,10,20,0.1); }
.upload-area { border: 2px dashed #C9A84C; border-radius: 10px; padding: 40px; margin-bottom: 20px; cursor: pointer; }
.upload-area:hover { background: #fdf8f0; }
input[type=file] { display: none; }
.btn { background: #3D0A14; color: #C9A84C; border: none; padding: 14px 40px; font-size: 1em; border-radius: 8px; cursor: pointer; width: 100%; margin-top: 10px; }
.btn:hover { background: #6B1A2A; }
.btn:disabled { background: #aaa; }
.error { background: #fdecea; color: #c0392b; padding: 15px; border-radius: 8px; margin-top: 20px; }
.result { margin-top: 30px; }
.result img { max-width: 100%; border-radius: 10px; }
footer { text-align: center; padding: 30px; color: #9a7a7a; font-size: 0.85em; }
</style>
</head>
<body>

<header>
<h1>Wine AI</h1>
<p>Valutazione Etichette Neuromarketing</p>
</header>

<div class="container">
<div class="card">

<h2>Analizza un'etichetta</h2>

<form id="uploadForm" method="POST" action="/analizza" enctype="multipart/form-data">

<div class="upload-area" onclick="document.getElementById('fileInput').click()">
<p>Clicca per scegliere un'immagine</p>
<p><small>.jpg / .jpeg / .png</small></p>

<input
    type="file"
    id="fileInput"
    name="immagine"
    accept="image/*"
    onchange="mostraAnteprima(this)"
> 
</div>

<img id="preview"
     src=""
     style="max-width:200px;display:none;margin:10px auto;border-radius:8px;">

<button type="submit"
        class="btn"
        id="submitBtn"
        disabled>
    Analizza Etichetta
</button>

</form>

{% if errore %}
<div class="error">
    Errore: {{ errore }}
</div>
{% endif %}

{% if report_path %}
<div class="result">
    <h3>Analisi completata!</h3>

    <img src="/report/{{ report_filename }}" alt="Report">

    <br><br>

    <a href="/report/{{ report_filename }}" download>
        <button class="btn" style="width:auto;">
            Scarica Report
        </button>
    </a>
</div>
{% endif %}

</div>
</div>

<footer>
Sistema AI locale - Progetto Universitario IULM
</footer>

<script>
function mostraAnteprima(input) {

    var preview = document.getElementById('preview');
    var btn = document.getElementById('submitBtn');

    if (input.files && input.files[0]) {

        var reader = new FileReader();

        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
            btn.disabled = false;
        };

        reader.readAsDataURL(input.files[0]);
    }
}
</script>

</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/analizza', methods=['POST'])
def analizza():

    if 'immagine' not in request.files:
        return render_template_string(
            HTML_TEMPLATE,
            errore="Nessun file caricato."
        )

    file = request.files['immagine']

    if file.filename == '':
        return render_template_string(
            HTML_TEMPLATE,
            errore="Nessun file selezionato."
        )

    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    try:
        sys.path.insert(
            0,
            os.path.dirname(os.path.abspath(__file__))
        )

        from analisi import genera_report

        report_path = genera_report(
            img_path,
            output_path='risultati/'
        )

        report_filename = os.path.basename(report_path)

        return render_template_string(
            HTML_TEMPLATE,
            report_path=report_path,
            report_filename=report_filename
        )

    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            errore=str(e)
        )


@app.route('/report/<filename>')
def serve_report(filename):

    return send_file(
        os.path.join('risultati', filename),
        mimetype='image/png'
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
