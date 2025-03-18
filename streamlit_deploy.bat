@echo off
echo Setting up repository for Streamlit deployment...

REM Remove all temporary files and scripts
echo Removing temporary files...
if exist temp_repo rmdir /S /Q temp_repo
del clean_and_push.bat
del exclude_list.txt
del final_push.bat
del reset_and_push.bat
del push_to_github.bat

REM Remove .git directory and recreate a fresh repository
echo Recreating Git repository...
rmdir /S /Q .git
"C:\Program Files\Git\cmd\git.exe" init

REM Configure Git
echo Configuring Git...
"C:\Program Files\Git\cmd\git.exe" config --local user.name "Nithin Adithya"
"C:\Program Files\Git\cmd\git.exe" config --local user.email "your-email@example.com"

REM Create .gitignore with correct entries
echo Updating .gitignore...
echo # Environment variables > .gitignore
echo .env >> .gitignore
echo >> .gitignore
echo # Database files >> .gitignore
echo *.db >> .gitignore
echo >> .gitignore
echo # Large data files >> .gitignore
echo Reviews.csv >> .gitignore
echo twitter_training.csv >> .gitignore
echo >> .gitignore
echo # Python >> .gitignore
echo __pycache__/ >> .gitignore
echo *.py[cod] >> .gitignore
echo *$py.class >> .gitignore
echo *.so >> .gitignore
echo .Python >> .gitignore
echo build/ >> .gitignore
echo develop-eggs/ >> .gitignore
echo dist/ >> .gitignore
echo downloads/ >> .gitignore
echo eggs/ >> .gitignore
echo .eggs/ >> .gitignore
echo lib/ >> .gitignore
echo lib64/ >> .gitignore
echo parts/ >> .gitignore
echo sdist/ >> .gitignore
echo var/ >> .gitignore
echo wheels/ >> .gitignore
echo *.egg-info/ >> .gitignore
echo .installed.cfg >> .gitignore
echo *.egg >> .gitignore
echo >> .gitignore
echo # Virtual Environment >> .gitignore
echo venv/ >> .gitignore
echo env/ >> .gitignore
echo ENV/ >> .gitignore
echo >> .gitignore
echo # Streamlit >> .gitignore
echo .streamlit/ >> .gitignore
echo >> .gitignore
echo # Temporary files >> .gitignore
echo *.bat >> .gitignore
echo !requirements.txt >> .gitignore
echo !streamlit_deploy.bat >> .gitignore
echo temp_repo/ >> .gitignore

REM Add everything to Git (excluding what's in .gitignore)
echo Adding files to Git...
"C:\Program Files\Git\cmd\git.exe" add .

REM Initial commit
echo Creating initial commit...
"C:\Program Files\Git\cmd\git.exe" commit -m "Initial commit for Streamlit deployment"

REM Set up remote
echo Setting up remote repository...
"C:\Program Files\Git\cmd\git.exe" remote add origin https://github.com/Nithin-Adithya/therapy-ai.git
"C:\Program Files\Git\cmd\git.exe" branch -M main

REM Push to GitHub
echo Pushing to GitHub...
"C:\Program Files\Git\cmd\git.exe" push -f origin main

echo Done!
echo Your code is now properly connected to GitHub and ready for Streamlit deployment.
echo Visit https://github.com/Nithin-Adithya/therapy-ai to verify.
pause 