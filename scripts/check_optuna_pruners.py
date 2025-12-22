import optuna
names = [n for n in dir(optuna.pruners) if 'ASHA' in n.upper()]
print('Matches:', names)
print('--- All non-private pruners:')
print('\n'.join([n for n in dir(optuna.pruners) if not n.startswith('_')]))
