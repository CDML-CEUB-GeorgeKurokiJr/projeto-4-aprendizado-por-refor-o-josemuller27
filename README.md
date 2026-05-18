# Projeto 4 — DCGAN aplicado ao CelebA

Trabalho da disciplina de Deep Learning. A proposta era pegar o exemplo clássico de DCGAN feito no MNIST (dígitos 28×28 em escala de cinza) e adaptar para algo mais desafiador: gerar **rostos humanos coloridos** a partir do dataset CelebA.

Notebook principal: `projeto-4-dcgan-dl.ipynb`
Notebooks de apoio (versões originais usadas como base): `MNIST_DCGAN.ipynb` e `CelebA_DCGAN.ipynb`.

---

## 1. Ideia geral

DCGAN (Deep Convolutional GAN) é uma variante da GAN clássica do Goodfellow que usa camadas convolucionais no lugar das densas. A intuição continua sendo a mesma de qualquer GAN — duas redes brigando entre si:

- O **gerador** parte de um vetor de ruído (100 números aleatórios) e tenta produzir uma imagem que pareça real.
- O **discriminador** recebe imagens reais e imagens geradas e tenta dizer quem é quem.

Treinando os dois juntos, o discriminador fica cada vez mais exigente, e o gerador é obrigado a melhorar pra continuar enganando ele. No fim, com sorte, o gerador aprende a representar a distribuição das imagens reais e produz rostos plausíveis a partir do nada.

Referências usadas:
- Goodfellow et al., *Generative Adversarial Nets* (2014).
- Radford, Metz & Chintala, *Unsupervised Representation Learning with Deep Convolutional GANs* (2015) — onde a arquitetura DCGAN é descrita.

---

## 2. O que mudou em relação ao MNIST

A migração não é só "trocar o dataset". Algumas decisões precisaram ser revistas:

| Aspecto         | MNIST                       | CelebA (este projeto)                |
|-----------------|-----------------------------|--------------------------------------|
| Canais          | 1 (escala de cinza)         | 3 (RGB)                              |
| Resolução       | 28×28                       | 64×64 (redimensionado)               |
| Tamanho do batch| 64                          | 128                                  |
| Épocas          | 20                          | 10 (mas cada época é muito maior)    |
| Carregamento    | `torchvision.datasets.MNIST`| `Dataset` customizado (pasta plana)  |

O dataset CelebA tem ~202 mil imagens, então mesmo com 10 épocas o gerador "vê" muito mais exemplos do que via no MNIST. Mantive `latent_dim = 100` porque é o valor padrão do paper e funciona bem na prática — aumentar não trouxe ganho perceptível em testes preliminares e só deixa o treino mais pesado.

---

## 3. Pré-requisitos e como rodar

O notebook foi pensado pra rodar no **Kaggle**, com a GPU ativada (T4 ou P100 servem). Local também funciona se você tiver uma GPU com pelo menos uns 6 GB de VRAM — em CPU é inviável (cada época levaria horas).

Passos no Kaggle:

1. Criar um Notebook novo.
2. Em *Data* → *Add Data*, adicionar `jessicali9530/celeba-dataset`.
3. Em *Settings* → *Accelerator*, selecionar GPU.
4. Subir o `projeto-4-dcgan-dl.ipynb` e rodar as células em ordem.

Dependências (todas já vêm no ambiente padrão do Kaggle):

```
torch >= 2.0
torchvision
Pillow
matplotlib
```

---

## 4. Pipeline de dados

O CelebA, quando baixado pelo Kaggle, vem como uma pasta plana de `.jpg` em:

```
/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/
```

Sem subpastas por classe, ou seja, `ImageFolder` não serve. Por isso escrevi um `Dataset` customizado bem simples:

```python
class CelebADataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = sorted([f for f in os.listdir(root) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.files[idx])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
```

As transformações fazem três coisas:

1. **Redimensionar para 64×64** (`Resize` + `CenterCrop`). As imagens originais são 178×218; cortar o centro mantém o rosto e descarta cabelo/fundo das laterais.
2. **Converter para tensor** com `ToTensor()` — valores passam pra `[0, 1]`.
3. **Normalizar para `[-1, 1]`** nos três canais. Esse intervalo é importante porque o gerador termina com `Tanh()`, que produz justamente valores em `[-1, 1]`. Se as imagens reais ficassem em `[0, 1]` e as falsas em `[-1, 1]`, o discriminador teria a vida fácil demais (separaria pela escala, não pelo conteúdo).

O `DataLoader` usa `batch_size=128`, `shuffle=True`, `num_workers=2` e `pin_memory=True` pra acelerar o transfer pra GPU.

---

## 5. Arquitetura do Gerador

A ideia é pegar um vetor de 100 números e crescer ele espacialmente até virar uma imagem 64×64 com 3 canais. Isso é feito com `ConvTranspose2d` (convolução transposta, às vezes chamada de "deconvolução"), que é basicamente um upsampling com filtros aprendidos.

Caminho de tamanhos:

```
(1×1)  →  (4×4)  →  (8×8)  →  (16×16)  →  (32×32)  →  (64×64)
 512        256       128         64           3
```

Cada bloco intermediário é `ConvTranspose2d → BatchNorm2d → ReLU`. A última camada substitui ReLU por `Tanh()` pra produzir saída em `[-1, 1]`.

Detalhe importante: o vetor latente chega como `(batch, 100)` mas precisa entrar na primeira `ConvTranspose2d` como `(batch, 100, 1, 1)`. Por isso o `view` no `forward`.

---

## 6. Arquitetura do Discriminador

Faz o caminho inverso: pega uma imagem `3×64×64` e vai reduzindo até virar um único número entre 0 e 1.

```
(64×64)  →  (32×32)  →  (16×16)  →  (8×8)  →  (4×4)  →  (1×1)
   3           64          128       256       512        1
```

Blocos: `Conv2d → BatchNorm2d → LeakyReLU(0.2)`. Duas coisas a notar:

- **A primeira camada não tem BatchNorm.** Isso é uma recomendação direta do paper do DCGAN — colocar BatchNorm na entrada da rede tende a desestabilizar o treino.
- **LeakyReLU em vez de ReLU.** No discriminador, ReLU "zera" os gradientes negativos e isso atrapalha. LeakyReLU com slope de 0.2 deixa um pouco de gradiente passar pelos dois lados.

A saída final passa por `Sigmoid()` (probabilidade de ser real) e é achatada pra `(batch, 1)`.

---

## 7. Inicialização de pesos

GANs são sensíveis à inicialização. O paper do DCGAN sugere `N(0, 0.02)` pra todas as camadas convolucionais, e isso é o que está implementado em `weights_init`:

- Camadas `Conv*`: peso normal com média 0 e desvio 0.02.
- Camadas `BatchNorm`: peso normal com média 1 e desvio 0.02; bias zerado.

Aplicado com `generator.apply(weights_init)` logo após instanciar as redes. Pode parecer um detalhe bobo, mas pular essa etapa costuma fazer o treino divergir nas primeiras épocas.

---

## 8. Função de perda e otimizadores

- **Perda:** `BCELoss` (binary cross-entropy). É a perda natural pra uma classificação binária real/falso.
- **Otimizador:** `Adam` com `lr = 0.0002` e `betas = (0.5, 0.999)`. O `beta1 = 0.5` (em vez do default 0.9) também vem do paper — reduzir esse parâmetro ajuda muito na estabilidade do treino adversarial.
- Tanto o gerador quanto o discriminador têm seu próprio otimizador.

Criei também um `fixed_noise` (um tensor `(64, 100)` aleatório, mas fixo) pra acompanhar visualmente como esses mesmos 64 vetores latentes vão sendo "decodificados" ao longo das épocas. É uma forma honesta de ver o progresso — se eu sorteasse ruído novo a cada época, não daria pra comparar.

---

## 9. Loop de treino

Para cada batch:

**Passo A — treinar o discriminador.**
1. Calcula a perda nas imagens reais (rótulo = 1).
2. Gera imagens falsas a partir de ruído e calcula a perda nelas (rótulo = 0). O `.detach()` evita que esse passo afete os gradientes do gerador.
3. Soma as duas perdas, faz `backward` e dá `step` no otimizador do D.

**Passo B — treinar o gerador.**
1. Passa as mesmas imagens falsas (sem `detach()` desta vez) pelo discriminador.
2. Calcula a perda como se elas fossem reais (rótulo = 1). A intuição é: "o gerador *quer* que o discriminador diga que essas imagens são reais; o quanto ele falhou nisso é o quanto o gerador precisa melhorar".
3. `backward` e `step` no otimizador do G.

A cada 200 batches imprimo o status. Ao fim de cada época, gero 64 amostras usando `fixed_noise` e salvo como PNG em `/kaggle/working/images_celeba/epoch_XX.png`. Isso vira o "filme" da evolução do gerador.

---

## 10. Resultados observados

O treino completo (10 épocas, ~202k imagens por época, batch 128 = 1583 batches por época) levou aproximadamente **40 minutos numa T4**.

Trecho dos logs no início e no fim:

```
Epoch [1/10] Batch    0/1583 Loss D: 1.6091  Loss G: 3.9220
Epoch [1/10] Batch  200/1583 Loss D: 0.4559  Loss G: 4.8882
...
Epoch [10/10] Batch 1200/1583 Loss D: 0.4447  Loss G: 1.4781
Epoch [10/10] Batch 1400/1583 Loss D: 0.5120  Loss G: 3.3006
```

A perda do discriminador estabilizou na faixa **0.3 — 0.7** e a do gerador oscila bastante entre **1 e 4**. Isso é saudável pra uma DCGAN: nenhuma das duas perdas vai pra zero (o que indicaria que um lado venceu), e elas continuam oscilando porque as redes ficam se ajustando uma à outra.

Visualmente:

- **Épocas 1–2:** rostos ainda muito borrados, mas já com a estrutura geral correta (manchas de pele no centro, cabelo escuro no topo, fundo mais claro).
- **Épocas 3–5:** olhos, boca e nariz aparecem em posições coerentes; tons de pele já variam.
- **Épocas 6–10:** os rostos ficam reconhecíveis. Alguns saem bem convincentes, outros têm artefatos clássicos de GAN (assimetria nos olhos, fundo "pastoso", oclusões esquisitas perto do cabelo).

Os arquivos `final_samples.png` e `loss_curves.png` ficam salvos em `/kaggle/working/` ao final do treino, junto com os pesos do gerador e do discriminador (`generator.pth` e `discriminator.pth`).

---

## 11. Problemas comuns e o que fazer

Coisas que me apareceram (ou que valem ficar de olho) durante o treino:

- **Mode collapse.** O gerador descobre uma imagem que sempre engana o discriminador e passa a só produzir variações disso. Sinal: as 64 amostras viram todas parecidas. Soluções: diminuir `lr` do gerador, treinar o D menos vezes, ou adicionar `label smoothing` (usar 0.9 em vez de 1.0 pros rótulos reais).
- **Perda explodindo.** Se em algum momento `Loss G` vira NaN ou cresce sem parar, geralmente é instabilidade numérica. Reduzir `lr` pra `1e-4` costuma resolver.
- **Discriminador dominando.** Se `Loss D` cai pra perto de 0, ele virou perfeito e o gerador não consegue mais aprender (gradientes zerados). Pode tornar o discriminador mais "fraco" (menos canais, ou treiná-lo a cada N batches em vez de todo batch).

Não cheguei a precisar usar nenhum desses ajustes — o treino padrão se mostrou estável o suficiente nas 10 épocas.

---

## 12. Estrutura do repositório

```
.
├── projeto-4-dcgan-dl.ipynb    # notebook principal (este projeto)
├── CelebA_DCGAN.ipynb          # versão de referência
├── MNIST_DCGAN.ipynb           # versão original em MNIST, ponto de partida
└── README.md                   # este arquivo
```

Saídas geradas em tempo de execução (no Kaggle, dentro de `/kaggle/working/`):

```
images_celeba/epoch_01.png ... epoch_10.png   # evolução do fixed_noise
final_samples.png                              # 64 rostos gerados ao final
loss_curves.png                                # curvas de perda G e D
generator.pth                                  # pesos do gerador
discriminator.pth                              # pesos do discriminador
```

---

## 13. Possíveis extensões

Algumas direções que ficaram em aberto, mas que seriam interessantes pra um trabalho seguinte:

- Subir a resolução pra 128×128 (precisa adicionar mais uma camada em cada rede).
- Trocar a perda BCE por **WGAN-GP** (Wasserstein loss com penalidade de gradiente), que é bem mais estável e pratimente elimina o problema de mode collapse.
- Fazer **interpolação no espaço latente**: pegar dois `z` quaisquer, gerar 10 pontos entre eles e ver a transição contínua entre os dois rostos. É um bom termômetro de quão "bem aprendido" o espaço latente está.
- Condicionar o gerador em atributos do CelebA (óculos, sorriso, cabelo loiro, etc.) — vira uma **cGAN** (Conditional GAN).
