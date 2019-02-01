# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 18:20:26 2019

@author: Sveta
"""

def main():

    while(1):
        print("Menu: ")
        print(">> 1. Train model")
        print(">> 2. Test")
        print(">> 0. Quit1")

        try:
            option = int(input("Select your option: "))
            print();
            if (option == 1):
                print("\nTraining in progress...")
                
                print("\nFinished!\n")
            elif(option == 2):
                print("\nTesting in progress...")
                
                print("\nFinished!\n")
            elif(option == 0):
                break
            else:
                print("Invalid option chosen")
                print("Please try again\n")
        except Exception as e:
            print('\nError occured')
            print('Please try again\n')


main()