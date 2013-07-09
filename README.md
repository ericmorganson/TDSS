TDSS
====

Software for the Time Domain Spectroscopic Survey



HOW TO USE GIT
====

TDSS github instructions:

There is a good help page at:
https://confluence.atlassian.com/display/STASH/Basic+Git+commands

1. Get a github account at:
https://github.com/

2. The TDSS repository is at:
https://github.com/ericmorganson/TDSS
You can get the “ssh clone URL” in the lower right hand corner

3. To actually get the repository, go to your unix directory where you want the repository and type:
git clone git@github.com:ericmorganson/TDSS.git
Where git@github.com:ericmorganson/TDSS.git is the “ssh clone URL”

4. Get the most recent update of the repository (not necessary immediately after the clone, but you should probably do it before you add or modify stuff)
git pull

5. Add a directory/file to repository:

# Making a directory and filling it with a script
cd TDSS
mkdir morganson
cp /a42185d1/morganson/LSD/ps1_new/python/make_lc.py morganson/.

# Adding the file to our local repository
git add morganson/make_lc.py

# Commit the file to your local repository (this is just local, you are not uploading anything) with a message ( -m )
git commit -m "adding morganson directory and a light curve maker"

# Push your files to the global repository (actually upload)
git push

6. The last two lines also work when you modify a file. This time I added a filename in git commit so that it only added that filename.
git commit morganson/make_lc.py -m "Added user wrapper to improve commandline ease."
git push
