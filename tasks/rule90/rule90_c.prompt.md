Write a simple, self-contained C program (under 40 lines of code) that implements a 1D Cellular Automaton using Rule 90 to generate a fractal pattern in the terminal.

The program should meet the following requirements:
* Use a fixed-width array of 64 cells and run for 32 generations.
* Start the simulation with a single active cell (represented by `1`) exactly in the middle of the array, with all other cells set to `0`.
* In each generation, print the array to the terminal using `#` for active cells and a space for inactive cells.
* Compute the next state of each cell using the bitwise XOR operator (`^`) on its left and right neighbors from the current generation.
* Use a secondary buffer array to safely calculate the next generation before updating the main array.

Avoid using complex if/else branching for the cell logic, keeping the code clean, efficient, and readable. Provide the source code wrapped cleanly inside a single ```c ... ``` block.
