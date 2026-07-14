from flask import Flask, render_template, request, session, redirect, url_for
import random

app = Flask(__name__)
app.secret_key = "chave_super_secreta_experimento"

# (Cole aqui os dicionários MAPA_CORRETO e ESTIMULOS_POR_CLASSE do seu código original)

@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

@app.route('/iniciar', methods=['POST'])
def iniciar():
    # Guarda o ID digitado pelo participante na sessão (substitui o simpledialog.askstring)
    session['id_participante'] = request.form.get('id_participante')
    
    # Inicializa o controlo das fases tal como fazia no Tkinter
    session['fase_atual_idx'] = 0
    session['bloco_atual_num'] = 1
    session['tentativa_atual_idx'] = 0
    session['acertos_bloco'] = 0
    session['dados_relatorio'] = []
    
    # Redireciona para a página principal do experimento
    return redirect(url_for('experimento'))

@app.route('/experimento')
def experimento():
    # Recupera em qual tentativa o usuário está (começa no 0)
    tentativa = session.get('tentativa_atual_idx', 0)
    
    # Se já chegou a 9 tentativas, vamos simular o fim do bloco
    if tentativa >= 9:
        return "<h1>Fim do bloco de teste! (O CSV será gerado nesta etapa no futuro)</h1>"

    # Para simularmos o jogo avançando, vamos mudar a letra do Modelo a cada tentativa!
    modelo_temporario = f"A{tentativa + 1}"

    return render_template('experimento.html',
                           fase="Pré-Teste AB",
                           modelo=modelo_temporario,
                           comparacoes=["B1", "B2", "B3"])

# --- NOVA ROTA: O OUVINTE DE RESPOSTAS ---
@app.route('/responder/<escolha>')
def responder(escolha):
    # Aqui o Python recebe exatamente em qual botão o seu amigo clicou (ex: 'B2')
    print(f"O participante escolheu: {escolha}")
    
    # Adicionamos +1 na tentativa atual e guardamos no "crachá" da sessão
    session['tentativa_atual_idx'] = session.get('tentativa_atual_idx', 0) + 1
    
    # Mandamos o navegador recarregar a tela do experimento com a próxima tentativa
    return redirect(url_for('experimento'))

if __name__ == '__main__':
    app.run(debug=True)