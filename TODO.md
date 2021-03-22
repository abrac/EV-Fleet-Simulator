- [ ] Automatically compress results with
      ```sh
      tar -c -I 'xz -9 -T10' -f 'Battery.out.csv.tar.xz' ./T*/*/Battery.out.csv
      ```
- [ ] Seperate common data-preprocessing tasks from `Kampala_UTX.py`.
- [ ] Make the package pip-installable.
