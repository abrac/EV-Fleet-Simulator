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

Please install the following software before installing `EV-Fleet-Sim`:

| Name       |    Version   | Description                                                                                                                                                                                                                                                                              |
|------------|:------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| *SUMO [^1] |   == 1.8.0   | Traffic mobility simulator. Once installed, make sure that the SUMO_HOME environment variable exists. [^0]                                                                                                                                                                               |
| *Python    |     > 3.8    | An awesome, easy-to-learn programming language named after a dangerous snake. [^5]                                                                                                                                                                                                       |
| *Git       |      Any     | A tool to collaborate on coding projects and track changes that we make.                                                                                                                                                                                                                 |
| *Bash      |      Any     | The terminal used in Linux and MacOS. Windows can emulate Bash using *Git Bash*. It will be automatically installed when you install Git.                                                                                                                                                |
| *SAM       | > 2020.11.29 | Simulator of Renewable-Energy Generators (Solar panels, wind-turbines, etc.).                                                                                                                                                                                                            |
| *R         |    > 4.0.4   | R programming language.                                                                                                                                                                                                                                                                  |
| *osmium    |    > 1.13    | Software for cropping OpenStreetMap files.                                                                                                                                                                                                                                               |
| *Pigz      |      Any     | Software for compressing files using multiple cpu threads. On Linux, install pigz with your package manager. On Windows, install pigz from [^2], and make sure it is in your system's PATH!! [^3] In MacOS, install MacPorts [^4], and then install Pigz using: `sudo port install pigz` |

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

| Name           | Version | Description                                        |
|----------------|:-------:|----------------------------------------------------|
| +gtfs2gps [^6] |  1.6-0  | Converts GTFS public transport data to GPS traces. |


Installing EV-Fleet-Sim
=======================

Once you've installed the dependencies, you can install EV-Fleet-Sim using the command:

```sh
pip install ev-fleet-sim
```

Note: If you develop with Python a lot, you should create a virtual environment for EV-Fleet-Sim, to protect its Python dependency versions from being changed by other Python softwares. If you don't understand a word of what I just said, you probably can ignore this.

[Optional] Creating a Virtual Environment for EV-Fleet-Sim
----------------------------------------------------------

Simply run the command `python -m venv <venv-path>`, where `<venv-path>` is the path where you would like to save your virtual environment. Unfortunately, you will need to activate the virtual environment every time you want to run EV-Fleet-Sim. Do this by running one of the following commands, depending on your platform:

| Platform    | Shell      | Command to activate virtual environment  |
|-------------|------------|------------------------------------------|
| Linux/MacOS | bash/zsh   | `source <venv-path>/bin/activate`        |
| Windows     | cmd.exe    | `C:\<venv-path>\Scripts\activate.bat`    |
|             | PowerShell | `PS C:\<venv-path>\Scripts\Activate.ps1` |

After you have activated the venv, you can install ev-fleet-sim into it with `pip install ev-fleet-sim`.

---

[^0]: See: https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html#sumo_home
      In MacOS, if you installed SUMO using Homebrew, the base directory of
      the SUMO installation is: "/opt/homebrew/share/sumo/"

[^1]: Make sure that libsumo is compiled with SUMO. The Ubuntu PPA does not 
      include it by default. Libsumo is required by *Step 4 (Routing)*. The 
      other steps will work without libsumo. I think the Windows binaries 
      released by sumo are compiled together with libsumo.

[^2]: https://blog.kowalczyk.info/software/pigz-for-windows.html

[^3]: https://zwbetz.com/how-to-add-a-binary-to-your-path-on-macos-linux-windows/#windows-gui

[^4]: https://www.macports.org/install.php

[^5]: Windows users: Make sure you tick the "Add Python to PATH" checkbox during 
      installation!!!

[^6]: Required if using GTFS data inputs. Install from 
      https://github.com/ipeaGIT/gtfs2gps/

