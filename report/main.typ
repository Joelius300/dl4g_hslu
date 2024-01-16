#import "template.typ": *

#show: project.with(
  title: "Jass Queen",
  authors: (
    "Joel L.",
  ),
)

= Zusammenfassung Bot Jass Queen

Der Code für das Projekt ist auf #link("https://github.com/Joelius300/dl4g_hslu")[GitHub (Joelius300/dl4g_hslu)] zu finden.

== Trumpfstrategie

Die verwendete Trumpfstrategie ist die Graf Heuristik @graf_jassen_nodate, die im Unterricht vorgestellt wurde. Sie erwies sich als die beste von allen evaluierten Ansätzen, darunter sind viele Variationen von Supervised Machine Learning. Mehr dazu steht in @supervised-trump geschrieben.

== Kartenstrategie

Die Strategie für die Wahl der Karte basiert auf Information Set Monte Carlo Tree Search (ISMCTS) @cowling_information_2012 mit Root Parallelization @cazenave_parallelization_nodate.

Da Jass über nicht-perfekte Informationen verfügt, wird bei ISMCTS immer als erstes ein vollständiger Spielzustand aus einem Information Set gesampled. Auf diesen unsicheren, aber perfekten Informationen können anschliessend die vier Schritte von Monte Carlo Tree Search, nämlich selection, expansion, rollout und backpropagation, durchgeführt werden. Knoten im Suchbaum, die nicht mit dem aktuellen Spielzustand (gesamplet vom Information Set) kompatibel sind, werden ignoriert. Am Ende wird der vielversprechendste Kindknoten des Wurzelknotens ausgewählt und gespielt.

Dieser bereits sehr starke Algorithmus kann auf verschiedene Arten parallelisiert werden @hutchison_parallel_2008. In diesem Projekt wird Root Parallelization umgesetzt, wobei mehrere Workers je einen Search Tree aufbauen und ISMCTS darauf betreiben. Am Ende werden alle Bäume zusammengeführt und die vielversprechendste Option über alle Bäume gesehen genommen. Solange die verwendeten Policies und Rollouts nicht vollständig deterministisch sind, erlaubt dies eine breitere Abdeckung des Suchraums, was zu einer genaueren Abschätzung der erwarteten Payoffs führt.

= Experimente

== Supervised Machine Learning für Trumpfstrategie <supervised-trump>

Um eine stärkere Trumpfstrategie als die Graf-Heuristik zu bauen, werden verschiedene Supervised Machine Learning Ansätze ausgearbeitet.

=== Evaluation

Ein wichtiger Punkt ist die Auswertung dieser Strategien. Es ist klar, dass nicht die Metriken auf den Daten verwendet werden dürfen, da diese nur aussagen, wie gut das Modell die Daten imitiert. Wie aber bereits am Fall von AlphaZero @silver_mastering_2017 gezeigt, kann es gut sein, dass eine neuartige Taktik, die tiefe Korrelation mit menschlichem Spiel aufweist, besser ist als diejenigen, die lediglich Menschen imitieren. Aus diesem Grund wird jede Trumpfstrategie wie folgt ausgewertet: Es werden 128 Jass Partien auf jeweils 1000 Punkte gegen die Graf Heuristik gespielt und die Gewinnrate berechnet. Die Kartenstrategie ist für alle Spieler ISMCTS. So kann aufgezeigt werden, ob diese Trumpfstrategie im Spiel tatsächlich die Überhand hat, was der Fall ist, wenn sie eine Gewinnrate von über 50% erzielt.

=== Methode

Die Experimente werden mit DVCLive @noauthor_dvclive_nodate getrackt. Dafür wird eine DVC Pipeline @noauthor_dvc_nodate erstellt, welche sowohl die Vorbereitung der Daten, die verschiedenen Trainingsschritte und die Evaluation beinhaltet. Während dem Training werden zudem Metriken wie der Loss und Klassifikationsmetriken wie Accuracy und F1 geloggt.

Die Daten werden vor der Verwendung mit undersampling balanciert, da Schieben gerade in den Swisslos Daten klar am häufigsten vorkommt. Normalisierung ist nicht nötig, da mit One-Hot Encoding gearbeitet wird und jedes Feature Binär (1 oder 0) ist.

=== Ansätze

Der einfachste Ansatz ist die alleinige Verwendung der Swisslos Daten. Dies erzielte eine Gewinnrate von 40% gegen Graf. Ein spannenderer Ansatz ist ein Pre-Training Ansatz, wo zuerst die Graf Heuristik gelernt wird (mit einem generierten Datensatz) und anschliessend nur mit den besten Spielern der Swisslos Daten ein Fine-Tuning gemacht wird.

- Training auf gesamten balancierten Swisslos Daten: 40% Gewinnrate gegen Graf
- Pre-Training auf Graf, Fine-Tuning auf Swisslos Daten (top 20%): 43% Gewinnrate gegen Graf

=== Modell

Das Modell nimmt die One-Hot kodierten Handkarten (36x1 binär) plus ob geschoben werden kann oder nicht (1x1 binär). Diese 37 Features werden in mehrere Linear Layers mit ReLU non-linearity gegeben. Bei jedem Layer werden die originalen Inputs hinzugefügt, um Skip-Connections zu ermöglichen. Am Ende ist ein Linear Layer, welcher die Logits für die 7 Trumpf Optionen (4 Farben + Une-ufe + Obe-abe + Schieben) ausgibt. Der verwendete Loss ist Cross-Entropy. Zusätzlich wird Dropout Regularisierung verwendet.

=== Tuning

Folgende Hyperparameter wurden ausgiebig getuned:

- Anzahl Layers
- Dimension der hidden Layers
- Learning rate
- Dropout rate
- Art/Implementation der Residual Connections
- Data
  - Nur Swisslos
  - Graf + Swisslos
  - Graf + Top 30% Swisslos
  - Graf + Top 20% Swisslos
  - Graf + Top 10% Swisslos

== Parameter Tuning für Monte Carlo Tree Search

Zwei Payoff Funktionen werden verglichen. Binärer Sieger (1 für Gewinner, 0 für Verlierer) schlägt punktebasierte Payoffs (tatsächliche Punkte auf [0,1] skaliert) nicht nur in Anzahl gewonnen Spielen sondern auch in Partien auf 1000 Punkte.

Der C Parameter für UCB1 wird ebenfalls getuned. Am Ende wurde 8 als beste Option gewählt, was erstaunlich war, da bisher $sqrt(2)$ sehr gut zu performen schien, aber es zeigt sich, dass $1 < sqrt(2) < 5 < 10 < 7 < 8$ gilt. Diese Tests werden im finalen Setting, das heisst auf dem Deployment Server mit Root Parallelized ISMCTS mit 15 Workers und 9.75 Sekunden Zeitbudget getestet.

== Kurzer Abstecher in Reinforcement Learning

Mit den Unterlagen zu Reinforcement Learning gelang es leider nicht einen Bot mit RL zu trainieren. Es wurde zwar versucht ein Gymnasium Environment @noauthor_gymnasium_nodate Wrapper aufbauend auf dem Jass-Kit GameSim zu erstellen, jedoch ohne Erfolg.

= Nennenswertes

Zusätzlich zum bisher genannten habe ich einige Dinge gemacht, die eventuell auch spannend sein könnten.
Diese können im Code angeschaut werden.

- Ausführliche ML Pipeline mit DVC
- Model Training mit PyTorch Lightning und DVCLive
- Generierung und Balancierung des Graf Datasets
- Verbesserungen/Anpassungen der jass-kit Bibliothek in Open-Source Fork (#link("https://github.com/Joelius300/jass-kit-py")[Joelius300/jass-kit-py])
- Evaluation von Agents auf mehreren Kernen
- Deployment mit Docker und Gunicorn

#bibliography("JassQueen.bib")
