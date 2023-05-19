---
title: Installation
---

Before installing EV-Fleet-Sim, you will need to install various software that EV-Fleet-Sim depends on.

Dependencies
============

> Note: Version numbers are just the ones that I tested the program with. Other 
> versions may also work. But many packages and software introduce breaking 
> changes between releases. So be aware of that.

> Note: Packages marked with a `*` are mandatory. The other packages are 
> recommended, but the software may work without them. Packages marked with a
> `+` are conditionally required. I.e., they are only required if performing 
> specific tasks. However, they are highly recommended.


Software
--------

Please install the following software before installing `EV-Fleet-Sim`. Install all the software with their default settings, except for Python on Windows.

**NB: Please install the correct versions of the software. Especially for SUMO and Python.**

| Name                                |                Version               | Description                                                                                                                                                                                                                                                                              |
|-------------------------------------|:------------------------------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| *[SUMO][1][^1]                     | == 1.8.0 <!-- **OR** 1.14.1[^7] --> | Traffic mobility simulator. Once installed, make sure that the SUMO_HOME environment variable exists.[^0]                                                                                                                                                                               |
| *[Python][2]                        |                == 3.9                | An awesome, easy-to-learn programming language named after a dangerous snake. **Windows users:** Make sure you tick the "Add Python to PATH" checkbox during installation!!![^5]                                                                                                        |
| *[Git][3]                           |                  Any                 | A tool to collaborate on coding projects and track changes that we make.                                                                                                                                                                                                                 |
| *Bash                               |                  Any                 | The terminal which comes with Linux and MacOS. Windows can emulate Bash using *Git Bash*. It will be automatically installed when you install Git.                                                                                                                                       |
| *[SAM][4]                           |             > 2020.11.29             | Simulator of Renewable-Energy Generators (Solar panels, wind-turbines, etc.).                                                                                                                                                                                                            |
| +[R][5]                             |                > 4.0.4               | R programming language. Only compulsory if you plan on running scenarios which use [GTFS data](https://gtfs.org/).                                                                                                                                                                       |
| +[osmium][6]                        |                > 1.13                | Software for cropping OpenStreetMap files. Compulsory on Linux and MacOS.                                                                                                                                                                                                                |
| +[osmconvert][7]                    |                  Any                 | Alternative software for cropping OpenStreetMap files. Compulsory on **Windows**.                                                                                                                                                                                                        |
| +[osmosis][8]                       |                  Any                 | Software used for overlaying NASA elevation data onto the OpenStreetMap files. Required if you want to consider elevation in the electric vehicle models of EV-Fleet-Sim.                                                                                                                |
| +[Microsoft Build Tools for C++][9] |                > 14.0                | Only compulsory on **Windows**. Used for compiling some Python packages.                                                                                                                                                                                                                 |

[1]: https://www.eclipse.org/sumo/
[2]: https://www.python.org/
[3]: http://git-scm.com/
[4]: https://sam.nrel.gov/
[5]: https://cran.r-project.org/
[6]: https://osmcode.org/osmium-tool/
[7]: https://wiki.openstreetmap.org/wiki/Osmconvert
[8]: https://wiki.openstreetmap.org/wiki/Osmosis
[9]: https://visualstudio.microsoft.com/visual-cpp-build-tools/
[12]: https://zwbetz.com/how-to-add-a-binary-to-your-path-on-macos-linux-windows/#windows-gui


SUMO Compilation Dependencies
-----------------------------

When compiling SUMO on Linux, you will need to install the following dependencies:

### Ubuntu

- cmake 
- python3 
- g++ 
- libxerces-c-dev 
- libfox-1.6-dev 
- libgdal-dev 
- libproj-dev 
- libgl2ps-dev 
- python3-dev 
- swig  <!-- I think... -->


Python Packages
---------------

When you install EV-Fleet-Sim with Pip, it will automatically install all the dependencies with their correct versions.


R packages
----------

| Name                | Version | Description                                        |
|---------------------|:-------:|----------------------------------------------------|
| +[gtfs2gps][1][^6] |  1.6-0  | Converts GTFS public transport data to GPS traces. |

[1]: https://github.com/ipeaGIT/gtfs2gps/

Installing EV-Fleet-Sim
=======================

Once you've installed the dependencies, you can install EV-Fleet-Sim using the command:

```sh
pip install ev-fleet-sim
```

Notes: 

* If you develop with Python a lot, you should create a virtual environment for EV-Fleet-Sim, to protect its Python dependency versions from being changed by other Python softwares. If you don't understand a word of what I just said, you probably can ignore this. For instructions, see the section: [Creating a Virtual Environment for  EV-Fleet-Sim](#venv).
* On Windows, please run this command and all other commands in this documentation from a `GIT Bash` terminal.


<details markdown='1' style="background:#EEEEEE;padding: 0.5em;">
<summary><a id=venv></a>Creating a Virtual Environment for  EV-Fleet-Sim [Optional]</summary><br>
Simply run the command `python -m venv <venv-path>`, where `<venv-path>` is the path where you would like to save your virtual environment. Unfortunately, you will need to activate the virtual environment every time you want to run EV-Fleet-Sim. Do this by running one of the following commands, depending on your platform:

| Platform    | Shell      | Command to activate virtual environment |
|-------------|------------|-----------------------------------------|
| Linux/MacOS | bash/zsh   | `source <venv-path>/bin/activate`       |
| Windows     | GIT Bash   | `source <venv-path>/Scripts/activate`   |

After you have activated the venv, you can install ev-fleet-sim into it with `pip install ev-fleet-sim`.

You can deactivate your virtual environment using the `deactivate` command.
</details><br>

Additional instructions for Windows
===================================

Unfortunately, Windows' search indexer slows down ev-fleet-sim drastically. In order to circumvent this, perform the following steps:

1. Open `Indexing Options` from the start menu. 
2. Click the `Advanced` button.
3. Switch to the `File Types` tab.
4. Un-check the `csv` and `xml` filetypes in the list.
    1. If they are not present in the list, you can add them using the `Add` button at the bottom of the window.
5. Click the `Ok` button.
6. If a `Rebuild Index` window pops up, press `Cancel`.
7. Finally, `Close` the `Indexing Options` window.

---

[^0]: See: [SUMO docs](https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html#sumo_home).

      In MacOS, if you installed SUMO using Homebrew, the base directory of
      the SUMO installation is: `/opt/homebrew/share/sumo/`

[^1]: Make sure that libsumo is compiled with SUMO. The Ubuntu PPA does not 
      include it by default. Libsumo is required by *Step 4 (Routing)*. The 
      other steps will work without libsumo. I think the Windows binaries 
      released by sumo are compiled together with libsumo.

[^5]: See image:
    
      ![Python Installer]({{site.baseurl}}/assets/images/docs/python_installation.png)

[^6]: Required if using GTFS data inputs. 
