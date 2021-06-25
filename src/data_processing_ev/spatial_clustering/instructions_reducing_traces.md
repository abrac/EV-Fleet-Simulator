Open each file in vim, and issue the following commands for each file:
```vim
:2,$v/,\([0-9]\|1[0-9]\)$/d OR
    :2,$v/,[0-9]$/d
:w
:bn
```
The command deletes all data-points with a velocity greater than 19 (1st command) OR 9 km/h (2nd command). 

This could be shortened with the following bash command (but for some reason it's not working):
```bash
nvim -c '2,$v/,\([0-9]\|1[0-9]\)$/d | w' *.csv
```
