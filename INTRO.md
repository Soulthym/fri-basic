# FRI: Implémentation basique en Rust et présentation théorique  
## Objectif  
Implémenter une version simplifiée du protocole FRI (Fast Reed-Solomon Interactive Oracle Proof of Proximity), l'une des composantes essentielles du protocole STARK, en langage Rust, et préparer une présentation théorique à présenter lors de l'entretien de suivi.  

## Directives  
### Implémentation :  

L'optimisation n'est pas nécessaire, privilégiez la clarté du code. Cela inclut l'absence d'implémentation de la Transformée de Fourier Rapide (FFT), Circle STARK etc...  
Pas besoin de prendre en compte le grinding.  

### Paramètres Fixes :  

Le facteur de repliement (folding factor) est fixé à 2. Vous n'avez pas besoin d'implémenter une solution générique prenant en charge d'autres facteurs de repliement.  

### Librairies Autorisées :  

Vous pouvez utiliser des librairies externes pour les corps finis (finite fields) et les arbres de Merkle (Merkle trees).  
Aucune autre librairie externe n'est autorisée.  

### Présentation Théorique :  
Préparez une courte présentation sous forme de slides sur le protocole FRI. Elle doit inclure une explication des concepts fondamentaux et de l'intérêt de FRI dans le contexte des protocoles STARK. Vous présenterez ces slides lors de l'entretien de suivi.  

## Livrables  
Le code source en Rust.  
Un fichier README expliquant les étapes pour compiler et exécuter votre implémentation.  
Une présentation en format PowerPoint, Google Slides, ou PDF que vous présenterez en direct lors de l'entretien de suivi.  

### Optionnel :  
des tests unitaires pour valider les principales fonctionnalités de votre implémentation Rust.  

## Critères d'évaluation  
Respect des consignes.  
Qualité et lisibilité du code.  
Pertinence, clarté et qualité visuelle de la présentation théorique.  
Compréhension des concepts fondamentaux de FRI et leur application dans le code.  

### Note :  
L'objectif est de comprendre non seulement votre capacité à étudier et implémenter des concepts cryptographiques complexes, mais aussi votre aptitude à expliquer, à justifier leur utilité et leur fonctionnement lors d'une présentation en direct.  
