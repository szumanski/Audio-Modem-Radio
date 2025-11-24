Este software so funciona em computadores por enquanto, futuramente vou tentar portar ele para Android

Para gerar o executavel voce deve primeiro instalar todas as bibliotecas nescessarias do projeto, depois execute o comando no terminal 
Recomendo Usar o Pycharm

pyinstaller --noconfirm --onefile --windowed --clean --name "FileBeepAdvanced" --hidden-import=sklearn.utils._cython_blas --hidden-import=sklearn.neighbors.typedefs --hidden-import=sklearn.neighbors.quad_tree --hidden-import=sklearn.tree --hidden-import=sklearn.ensemble --collect-all sklearn --collect-all scipy filebeep_advanced_v2.py

Manual Completo do FileBeep Advanced v2
📋 Índice
1.	Introdução
2.	Instalação e Primeiros Passos
3.	Aba "📤 Transmitir"
4.	Aba "📥 Receber"
5.	Aba "📊 Monitor"
6.	Aba "⚙️ Configurações"
7.	Dicas e Melhores Práticas
8.	Solução de Problemas
   
🎯 Introdução
O FileBeep Advanced v2 é um sistema profissional de transferência de arquivos via áudio que permite enviar qualquer tipo de arquivo (documentos, imagens, vídeos, etc.) usando som como meio de transmissão. É ideal para situações onde não há conexão de internet ou rede disponível.
Como Funciona?
•	Transmissão: Converte arquivos em sinais sonoros especiais
•	Recepção: Captura o áudio e reconverte para arquivo original
•	Recursos Avançados: Compressão inteligente, correção de erros, múltiplos modos de transmissão

🚀 Instalação e Primeiros Passos
Requisitos do Sistema
•	Windows, macOS ou Linux
•	Alto-falantes funcionais
•	Microfone (para recepção)
•	100MB de espaço livre
Primeira Execução
1.	Execute o arquivo filebeep_advanced_v2.py
2.	Aguarde o carregamento da interface
3.	Verifique se o áudio está funcionando
   
📤 Aba "Transmitir"
🎛️ Configurações de Transmissão
Modulação (Método de Transmissão)
•	FSK1200: Lenta e robusta - Ideal para condições ruins
•	FSK9600: Equilibrada - Bom para uso geral ✓
•	BPSK: Robusta - Boa imunidade a ruído
•	QPSK: Eficiente - 2x velocidade do BPSK
•	8PSK: Rápida - 3x velocidade do BPSK
•	FSK19200: Alta velocidade - Para bons canais
•	OFDM4/8: Avançada - Resistente a interferências
•	SSTV: Para imagens - Velocidade muito lenta
•	HELLSCHREIBER: Para texto - Estilo fac-símile
Taxa de Símbolo
•	Controla a velocidade de transmissão
•	600-38400 símbolos por segundo
•	Recomendado: 9600 para uso geral
Compressão Ativa ✓
•	Reduz o tamanho dos arquivos antes de enviar
•	Sempre ativada para melhor desempenho

📁 Ações de Transmissão
🔒 Codificar Arquivo Único
•	Para arquivos de até 5MB
•	Gera um único arquivo de áudio
•	Processamento rápido
📦 Codificar Arquivo Grande (Multi-partes)
•	Para arquivos maiores que 5MB
•	Divide em partes menores
•	Cada parte é um arquivo de áudio separado
•	Duração da Parte: 1-10 minutos (recomendado 2-3)

🎵 Player de Transmissão
Lista de Reprodução
•	Mostra todos os arquivos gerados
•	Cores:
o	🔴 Vermelho: Não reproduzido
o	🟡 Amarelo: Reproduzindo
o	🟢 Verde: Reprodução concluída
Controles do Player
•	▶️ Reproduzir: Toca o arquivo selecionado
•	⏸️ Pausar: Pausa a reprodução atual
•	⏹️ Parar: Para completamente
•	🗑️ Limpar Lista: Remove todos os arquivos da lista

📊 Estatísticas da Transmissão
•	Tamanho do arquivo: Tamanho original
•	Tempo estimado: Duração da transmissão
•	Eficiência: Velocidade em bytes/segundo

📥 Aba "Receber"
⚙️ Configurações de Recepção
Modulação
•	DEVE SER IGUAL à usada na transmissão
•	Se não souber, teste com QPSK ou FSK9600
Taxa de Símbolo
•	DEVE SER IGUAL à usada na transmissão

🎤 Controles de Recepção
Iniciar Recepção
•	Grava áudio do microfone por 5 minutos
•	Decodifica automaticamente os arquivos
•	Mostra progresso em tempo real
Decodificar de Arquivo WAV
•	Para arquivos de áudio pré-gravados
•	Selecionar arquivo .WAV para decodificar

📈 Nível de Entrada
•	Medidor de Volume: Mostra o volume captado
•	Ideal: Manter entre 30-70%
•	Muito baixo: Aumente o volume da fonte
•	Muito alto: Reduza o volume para evitar distorção

🔄 Status de Montagem
•	Mostra progresso de arquivos multi-partes
•	Indica partes faltantes
•	Atualização automática a cada 2 segundos

📂 Arquivos Recebidos
•	Lista todos os arquivos decodificados com sucesso
•	Organizados por data e hora

📊 Aba "Monitor"
📈 Métricas em Tempo Real
•	Taxa de bits: Velocidade atual
•	SNR: Qualidade do sinal (quanto maior, melhor)
•	BER: Taxa de erro (quanto menor, melhor)
•	Qualidade: Avaliação geral

📝 Log de Atividades
•	Registro completo de todas as operações
•	Limpar Log: Apaga o histórico atual
•	Salvar Log: Guarda em arquivo para análise

⚙️ Aba "Configurações"
🔊 Configurações de Áudio
Taxa de Amostragem
•	44100 Hz: Qualidade padrão
•	48000 Hz: Qualidade melhor
•	96000 Hz: Alta qualidade ✓
Dispositivo de Áudio
•	Seleciona alto-falantes para transmissão
•	Usa dispositivo padrão do sistema

💾 Configurações de Arquivo
Diretório de Cache
•	Pasta onde arquivos temporários são guardados
•	Procurar...: Selecionar nova pasta
Limpar cache automaticamente ✓
•	Remove arquivos temporários automaticamente

🛠️ Ações do Sistema
🧹 Limpar Cache
•	Remove todos os arquivos temporários
•	Libera espaço em disco

🔄 Restaurar Padrões
•	Volta todas as configurações para o padrão
•	Não afeta arquivos recebidos

💡 Dicas e Melhores Práticas
✅ Para Melhor Qualidade
1.	Ambiente Silencioso o	Evite ruídos de fundo o	Feche janelas e portas
2.	Posicionamento Ideal o	Alto-falante e microfone próximos o	Mas não encostados (evite feedback)
3.	Configurações Recomendadas text
Modulação: QPSK ou FSK9600
Taxa de Símbolo: 9600
Taxa de Amostragem: 96000 Hz
Compressão: ATIVADA
4.	Para Arquivos Grandes
o	Use divisão em partes de 2-3 minutos
o	Verifique cada parte individualmente

⚠️ O Que Evitar
•	Nunca mude a modulação durante a transmissão
•	Não mova o microfone durante a recepção
•	Evite superfície que vibram (mesa instável)
•	Não use volume máximo (causa distorção)

🔧 Solução de Problemas
❌ Problemas Comuns
"Nenhum Arquivo Decodificado"
1.	Verifique se a modulação está correta
2.	Aumente o volume da fonte
3.	Teste em ambiente mais silencioso
4.	Verifique se o microfone está funcionando
"Arquivo Corrompido"
1.	Retransmita com modulação mais lenta (FSK1200)
2.	Reduza a taxa de símbolo
3.	Verifique conexões de áudio
"Player Não Reproduz"
1.	Verifique se o áudio do sistema funciona
2.	Teste com outro arquivo de áudio
3.	Reinicie o programa
"Recepção Muito Lenta"
1.	Use modulação mais rápida (8PSK, OFDM)
2.	Aumente a taxa de símbolo
3.	Verifique a qualidade do áudio
   
📞 Suporte
•	Verifique o log de atividades para detalhes técnicos
•	Teste sempre com arquivos pequenos primeiro
•	Use modulações mais simples para teste inicial

🎉 Parabéns!
Agora você está pronto para usar o FileBeep Advanced v2 como um profissional! Comece com arquivos pequenos e testes simples, depois avance para transmissões mais complexas.
Lembre-se: A prática leva à perfeição. Cada ambiente é único e pode requerer ajustes específicos.
