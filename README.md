# quant-risk-engine

Estimativa de risco para uma série de retornos: média, desvio padrão amostral e Value at Risk (VaR) paramétrico gaussiano de um período.

O código fica em [`quant-risk-engine/`](quant-risk-engine/). `analyzer.py` concentra o cálculo. `example.py` simula um ano de retornos diários e imprime um relatório. A série de entrada é de retornos ou de P&L, não de preços.

## Stack tecnológico

- Python 3
- NumPy
- Pandas
- Pytest

Nenhuma outra dependência é necessária. O quantil da normal padrão é calculado no próprio módulo.

## Instalação no Ubuntu / WSL

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
cd quant-risk-engine
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

O pacote `python3-venv` é obrigatório no Ubuntu e no WSL. Sem ele, `python3 -m venv` falha. Ative o ambiente em cada terminal novo com `source venv/bin/activate` antes de usar `pytest` ou `python`.

## Testes

```bash
cd quant-risk-engine
source venv/bin/activate
pytest -v
```

## Exemplo

```bash
cd quant-risk-engine
source venv/bin/activate
python example.py
```

O script usa a semente `42`, drift anual de 8% e volatilidade anual de 20%. O resultado é reproduzível.

## VaR paramétrico

Seja \(R_1, \ldots, R_n\) a série de retornos. A média e o desvio padrão amostral são

\[
\hat\mu = \frac{1}{n}\sum_{i=1}^{n} R_i
\qquad
\hat\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(R_i-\hat\mu)^2}.
\]

Sob \(R \sim \mathcal{N}(\mu, \sigma^2)\), o VaR de um período ao nível de confiança \(c\) é a perda no quantil esquerdo da distribuição:

\[
\mathrm{VaR}_c = -\hat\mu + z_c\,\hat\sigma,
\qquad
z_c = \Phi^{-1}(c).
\]

\(\Phi^{-1}\) é a inversa da distribuição normal padrão. Para \(c = 0{,}95\), \(z_{0{,}95} \approx 1{,}64485\).

Um VaR de `0,012` significa que, se a hipótese normal for adequada, a perda do período só supera 1,2% da posição em cerca de \(1-c\) dos casos. O sinal positivo representa perda em relação a zero, não em relação à média.
