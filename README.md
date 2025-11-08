# Genetic Algorithm for Penguin Classification using Perceptron

## Overview
This project demonstrates how to use a **Perceptron**, a simple linear classifier, to classify penguin species (Adelie and Gentoo) based on various physical attributes using a **Genetic Algorithm (GA)**.  
The original problem uses the **Iris dataset** and is modified here to use the **Penguin dataset** obtained from the UCI Machine Learning Repository.

---

## Problem
The goal is to predict the species of penguins based on the following features:
- Bill Length (mm)
- Bill Depth (mm)
- Flipper Length (mm)
- Body Mass (g)

This problem is a **binary classification problem** that uses **Perceptron** (a type of neural network) and **Genetic Algorithm (GA)** to optimize the classification.

---

## How It Works
### Step 1: **Perceptron Model**
A **Perceptron** model is used to classify two types of penguins: **Adelie** and **Gentoo**. This model learns by adjusting weights based on prediction errors in the training data.

### Step 2: **Genetic Algorithm**
The **Genetic Algorithm** optimizes the **weights** and **biases** of the perceptron by:
1. Creating an initial population of random weights.
2. Using **selection, crossover, and mutation** to generate new populations.
3. Iteratively improving solutions across multiple generations.
4. Stopping when an optimal or near-optimal solution is found.

### Step 3: **Model Training and Evaluation**
The model is trained on the **Penguin dataset**, and the perceptron is evaluated for its **accuracy** on the test set.

---

