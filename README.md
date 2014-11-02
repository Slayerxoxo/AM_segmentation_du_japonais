AM_segmentation_du_japonais
===========================

Script permettant de segmenter en mot le japonais.

Réalisé dans le cadre du cours "Applications Multilingues", en M2 ATAL(Apprentissage et Traitement Automatique de la Langue) de l'Université de Nantes.

Date de rendu : 3 novembre 2014


==========================

# Commande pour construire les ensembles de train/test à partir du KNBC
python knbc_to_xml.py knbc-train.xml knbc-test.xml knbc-reference.xml

# Commande pour segmenter le fichier de test avec mon implémentation de hmm
python hmm_segmenter.py knbc-train.xml knbc-test.xml knbc-hmm.xml

# Commande pour évaluer la performance d'un système
python evaluation.py knbc-hmm.xml knbc-reference.xml

Meilleurs résultats : 

Avg Precision 0.957560380247
Avg Recall 0.934352408883
Avg f-measure 0.945814049226

