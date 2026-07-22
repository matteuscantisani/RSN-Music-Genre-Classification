# RSN-Music-Genre-Classification

Sistema para **classificação automática de gêneros musicais** utilizando técnicas de Processamento Digital de Sinais (DSP) e Aprendizado de Máquina/Deep Learning.

O projeto contempla desde a construção do dataset até o treinamento e avaliação de modelos capazes de identificar o gênero de um trecho de áudio.

---

# Objetivos

* Construir um pipeline automatizado para processamento de áudio.
* Criar uma extensão do dataset GTZAN com gêneros brasileiros.
* Extrair características acústicas dos áudios.
* Treinar e avaliar modelos de classificação de gêneros musicais.
* Comparar diferentes arquiteturas de aprendizado de máquina.

---

# Gêneros Musicais

## GTZAN Original

* Blues
* Classical
* Country
* Disco
* Hip-Hop
* Jazz
* Metal
* Pop
* Reggae
* Rock

## Gêneros Brasileiros

* Samba/Pagode
* Forró
* Funk

---

# Pipeline para construção do dataset

O dataset é montado em 6 etapas, executando os scripts em `data_loader/scripts/` em ordem:

| Script | Etapa |
|---|---|
| `00_download.py` | Download dos zips do HuggingFace |
| `01_extract.py` | Extração dos áudios compactados |
| `02_convert.py` | Conversão para WAV 22.050 Hz mono PCM 16 bits |
| `03_select_segment.py` | Seleção do trecho mais representativo (30s) |
| `04_build_dataset.py` | Organização do dataset por gênero |
| `05_add_gtzan.py` | Integração com GTZan |

Para executar tudo de uma vez:

```bash
python3 baixar_dataset.py
```

---

# Tecnologias

* Python
* Librosa
* NumPy
* Pandas
* SciPy
* Pydub
* SoundFile
* TensorFlow (modelos)
* Matplotlib

---

# Dataset

O projeto utiliza como base o **GTZAN Music Genre Dataset**, complementado com três gêneros brasileiros para fins de pesquisa.

Cada amostra possui:

* duração de 30 segundos;
* frequência de amostragem de 22.050 Hz;
* áudio mono;
* formato WAV.

---


# Requisitos

Instale as dependências com:

```bash
pip install -r requirements.txt
```

---

# Autores

- [Matteus Marinho Cantisani](https://github.com/matteuscantisani)
- [Lucas Araújo Medeiros](https://github.com/lucasaraujomedeiros)
- [Júlia Cruz de Araújo Andrade](https://github.com/juliacruzandrade)
- [José Paulo Freitas da Silva Farias](https://github.com/josepaulofsf)

Universidade Federal de Campina Grande (UFCG)

Projeto de pesquisa em Classificação Automática de Gêneros Musicais.
