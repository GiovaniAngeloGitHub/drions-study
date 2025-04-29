# DRIONS Study - Segmentação do Disco Óptico com U-Net++ em PyTorch (CPU-only)

Este projeto realiza **segmentação do disco óptico** em imagens de **retinografia** da base **DRIONS-DB** usando **U-Net++ com EfficientNet-b0** como backbone, implementado com `segmentation-models-pytorch`.

> **Objetivo**: identificar com precisão a região do disco óptico em imagens fundoscópicas, mesmo com base pequena e uso de CPU.

---

## 📂 Como obter a base de dados DRIONS-DB

A base pode ser baixada gratuitamente em:

🔗 **[DRIONS-DB - Universidad de Alicante](https://www.ua.es/en/servicios/scie/base-de-datos-drions-db.html)**

Após o download:

1. Extraia os arquivos `.jpg` para:
   ```
   data/images/
   ```

2. Copie as **máscaras geradas pelos especialistas** (ou gere a partir dos arquivos de anotação `.txt`) para:
   ```
   data/masks/
   ```

3. Coloque os arquivos de anotação `.txt` dos especialistas (ex: `anotExpert1_001.txt`) em:
   ```
   data/experts_annotation/
   ```

> O projeto pressupõe que as imagens e máscaras tenham nomes compatíveis, ex: `image_001.jpg` e `image_001.png`.

---

## 📁 Estrutura de Diretórios

```
drions-study/
├── data/
│   ├── images/                # Imagens originais da retina (JPG)
│   ├── masks/                 # Máscaras binárias do disco óptico (PNG)
│   └── experts_annotation/    # Arquivos .txt com coordenadas dos especialistas
│
├── outputs/
│   ├── predictions/           # Máscaras geradas pelo modelo
│   └── comparisons/           # Comparações visuais entre predição, ground-truth e erro
│
├── src/
│   ├── main.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── best_model.pth             # (gerado) modelo com melhor desempenho no Dice
├── README.md
└── pyproject.toml
```

---

## ⚙️ Requisitos e instalação

Este projeto foi desenvolvido para rodar **em CPU apenas** e depende do gerenciador de pacotes **[uv](https://github.com/astral-sh/uv)**.

### ✅ Pré-requisitos

- Python >= 3.10
- `uv` instalado ([guia oficial](https://github.com/astral-sh/uv))

### 🔧 Instalação com `uv`

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r pyproject.toml
```

---

## 🚀 Como executar

Execute o pipeline completo (treinamento + avaliação + geração de figuras):

```bash
uv run src/main.py
```

- Se `best_model.pth` já existir, ele será carregado.
- Caso contrário, o modelo será treinado do zero (com 50 épocas).
- Ao final, os seguintes arquivos são gerados:

  - `outputs/predictions/*.png` → máscaras previstas
  - `outputs/comparisons/*.png` → imagem da retina com:
    - ground-truth (verde)
    - predição (vermelho)
    - erro/diferença (azul)

---

## 📈 Avaliação

A performance do modelo é avaliada com a **métrica Dice** no conjunto de validação.  
A visualização dos erros permite analisar onde o modelo acerta, erra ou falha parcialmente.

---

## 📜 Licença

Este projeto é apenas para fins acadêmicos e educacionais.  
A base de dados DRIONS-DB é de propriedade da Universidad de Alicante e deve ser usada conforme os termos de uso da instituição.
