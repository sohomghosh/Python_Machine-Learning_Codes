#force reinstall packages in python
pip install --upgrade --force-reinstall

#Trace what's happening wrong
strace python -c "import talib"
