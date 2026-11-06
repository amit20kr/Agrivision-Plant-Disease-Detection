import json

with open('eval_report/eval_test.json') as f:
    clean = json.load(f)
with open('eval_report/eval_test_corrupt.json') as f:
    corrupt = json.load(f)

print('=== CLEAN ===')
print('accuracy:', clean['keras']['accuracy'])
print('n_samples:', clean['keras']['n_samples'])
rep = clean['keras']['report']
for cls, vals in rep.items():
    if cls not in ['accuracy','macro avg','weighted avg']:
        print(f"  {cls}: F1={vals['f1-score']:.3f}, P={vals['precision']:.3f}, R={vals['recall']:.3f}")

print()
print('=== CORRUPT ===')
print('accuracy:', corrupt['keras']['accuracy'])
print('n_samples:', corrupt['keras']['n_samples'])
