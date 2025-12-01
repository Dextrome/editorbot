from src.audio.demucs_wrapper import DemucsSeparator

s = DemucsSeparator()
print('Starting lightweight demucs wrapper test...')
try:
    s.separate('non_existent_file_hopefully_missing_12345.wav')
except Exception as e:
    print('Caught exception type:', type(e).__name__)
    print('Exception message:', e)
    import traceback
    traceback.print_exc()
