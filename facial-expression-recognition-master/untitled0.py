# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 06:34:38 2017

@author: harry
"""
scorecard = []

i = 0
x = 0
y = 0
while i < 1.0 :
    scorecard[x] = []
    i = i + 0.05
    j = 10
    scorecard[x][y] = i , j
    #print (i)
    x += 1 

print (scorecard)