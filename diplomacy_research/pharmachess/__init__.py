"""PharmaChess: Polypharmacy Risk as a Metabolic Territory Control Game

Adapts the Richelieu self-evolving multi-agent framework (NeurIPS 2024) to
drug safety.  The human body's CYP enzyme / transporter network is the game
board; each drug in a polypharmacy regimen is a "power" that occupies and
competes for metabolic territories.  An LLM-based coordinator synthesises
evidence from FAERS, DrugBank and PubMed and a self-play loop augments its
memory bank without requiring any human-labelled training data.
"""
