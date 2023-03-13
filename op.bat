@echo off

if "%1" == "a" (
  call .envr\Scripts\activate.bat
) else if "%1" == "d" (
  call .envr\Scripts\deactivate.bat
) else (
  echo Invalid option. Please use 'a' to activate or 'd' to deactivate the virtual environment.
)