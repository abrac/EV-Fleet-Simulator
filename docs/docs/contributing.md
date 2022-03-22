---
title: Contibuting
---

Now I'll show you how to make changes to the program, and how to upload your changes so that the whole community can benefit from them.

Let's imagine that you want to add some new functionality to the program. You want it to greet us with "Hello world!" when we run `ev-fleet-sim`.

1. Create a folder for storing EV-Fleet-Sim's source code.

2. Clone the source code from the internet to this folder. We will calll this folder's directory `<source-dir>`.

   ```sh
   git clone https://gitlab.com/eputs/ev-fleet-sim.git <source-dir> # Clone the git repository into the new folder.
   ```

1. Open `<source-dir>/src/data_processing_ev/__init__.py` in your in your favourite text-editor.

3. Go down to the line which says:

    ```python
    def main():
    ```

4. If that is line `x`, insert in line `x+1`:

    ```python
    print("Hello world!")
    ```

5. Save the file, open a terminal, run `pip install <source-dir>`, and then run `ev-fleet-sim`. It should say "Hello
   world!" before continuing with the program.

You will see that `__init__.py` defines a function called `run()`. This function is
responsible for running the *sub-modules* corresponding each of the steps.

If you want to edit the functionality of a particular step, you have to open
the module corresponding to that step. Each module is contained in a folder in
the `<source-dir>/src/data_processing_ev` folder. Simply open the folder of that module
and edit its python scripts.

If you have made any changes that are worth contributing back to the main
project, we would gratefully accept them. This is how you do it:

1. Open the EV-Fleet-Sim repository [in GitLab](https://gitlab.com/eputs/ev-fleet-sim/), and press the *Fork* button.

2. Clone your fork to your PC by running the following command in a terminal (replacing `<user-name>` and `<source-dir-fork>`)

   ```sh
   git clone https://gitlab.com/<user-name>/ev-fleet-sim.git <source-dir-fork>
   ```

   and then make the changes that you want to make on your fork.


3. *Commit* the changes to the local copy of your fork, using the following commands.

   ```sh
   git add <file-you-changed>
   git add <another-file-you-changed>
   git commit -m <a-short-summary-of-the-changes>
   ```

3. Push the commits to the online copy of your fork.

   ```sh
   git push
   ```

4. Open your fork's repository in GitLab and [create a *merge request*](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html#when-you-work-in-a-fork). This will notify me that you would like to send some changes (commits) to me. With a few clicks I will be able to accept the changes that I like, and reject the changes that I don't.

Why don't you create a fork, do this Hello-World exercise on your fork, and create a pull-request. Of course, I will reject the pull-request, but consider it the "initiation rites" of EV-Fleet-Sim 😉.

Also remember to press the "star" and "notification bell" on the top of the [GitLab page](https://gitlab.com/eputs/ev-fleet-sim). That way, more people will be made aware of the software.