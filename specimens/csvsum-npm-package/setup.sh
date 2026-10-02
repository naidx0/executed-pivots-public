mkdir -p /app/csvsum/src /app/csvsum/bin /app/csvsum/test /data
# npm is offline on this machine: no registry, no audit, no update check
cat > /root/.npmrc <<'EOF'
offline=true
audit=false
fund=false
update-notifier=false
EOF
cat > /app/csvsum/package.json <<'EOF'
{
  "name": "csvsum",
  "version": "0.3.0",
  "description": "Summarize one numeric column of a CSV file",
  "type": "module",
  "main": "lib/index.js",
  "bin": {
    "csvsum": "bin/csvsum"
  },
  "files": [
    "src"
  ],
  "scripts": {
    "test": "node --test test/"
  },
  "license": "MIT"
}
EOF
cat > /app/csvsum/src/index.js <<'EOF'
'use strict';
const fs = require('fs');

// Parse a small CSV (no quoting) into row objects keyed by the header.
function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== '');
  if (lines.length === 0) return { header: [], rows: [] };
  const header = lines.shift().split(',').map((h) => h.trim());
  const rows = lines.map((l) => {
    const cells = l.split(',');
    const row = {};
    header.forEach((h, i) => {
      row[h] = (cells[i] || '').trim();
    });
    return row;
  });
  return { header, rows };
}

// count, total, mean, min and max of one column; empty cells are skipped
function summarize(text, column = 'amount') {
  const { header, rows } = parseCsv(text);
  if (!header.includes(column)) throw new Error(`no column ${column}`);
  let total = 0;
  let count = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const row of rows) {
    const v = row[column];
    if (v === '') continue;
    total += v;
    count += 1;
    min = Math.min(min, v);
    max = Math.max(max, v);
  }
  if (count === 0) return { count: 0, total: 0, mean: 0, min: 0, max: 0 };
  return { count, total, mean: total / count, min, max };
}

function format(s) {
  const f = (x) => Number(x).toFixed(2);
  return `count=${s.count} total=${f(s.total)} mean=${f(s.mean)} min=${f(s.min)} max=${f(s.max)}`;
}

function summarizeFile(path, column) {
  return summarize(fs.readFileSync(path, 'utf8'), column);
}

module.exports = { parseCsv, summarize, format, summarizeFile };
EOF
cat > /app/csvsum/bin/csvsum.js <<'EOF'
'use strict';
const { summarizeFile, format } = require('../src/index.js');

const [file, column = 'amount'] = process.argv.slice(2);
if (!file) {
  console.error('usage: csvsum FILE [COLUMN]');
  process.exit(2);
}
try {
  console.log(format(summarizeFile(file, column)));
} catch (e) {
  console.error(`csvsum: ${e.message}`);
  process.exit(1);
}
EOF
cat > /app/csvsum/test/index.test.js <<'EOF'
'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { parseCsv, summarize, format } = require('../src/index.js');

test('parseCsv keys rows by the header', () => {
  const { header, rows } = parseCsv('id, amount\n1, 2.5\n\n2,3\n');
  assert.deepStrictEqual(header, ['id', 'amount']);
  assert.deepStrictEqual(rows, [{ id: '1', amount: '2.5' }, { id: '2', amount: '3' }]);
});

test('summarize adds numbers, not strings', () => {
  const s = summarize('id,amount\n1,12.50\n2,3\n3,4.5\n');
  assert.strictEqual(s.count, 3);
  assert.strictEqual(s.total, 20);
  assert.strictEqual(s.min, 3);
  assert.strictEqual(s.max, 12.5);
});

test('summarize skips empty cells and reads other columns', () => {
  const s = summarize('id,amount,fee\n1,10,0.5\n2,,1.5\n3,-4,\n', 'fee');
  assert.deepStrictEqual(s, { count: 2, total: 2, mean: 1, min: 0.5, max: 1.5 });
});

test('format prints two decimals', () => {
  assert.strictEqual(format({ count: 2, total: 7, mean: 3.5, min: -1, max: 8 }),
    'count=2 total=7.00 mean=3.50 min=-1.00 max=8.00');
});

test('unknown column is an error', () => {
  assert.throws(() => summarize('id,amount\n1,2\n', 'price'), /no column price/);
});
EOF
cat > /app/csvsum/README.md <<'EOF'
# csvsum

Summarize one numeric column of a CSV file (header row, no quoting).

    csvsum FILE [COLUMN]      # COLUMN defaults to "amount"

prints `count=N total=T mean=M min=A max=B` with two decimals. Empty cells are skipped.

Library use: `const { summarize, format } = require('csvsum')`.

Development: `npm test`. Install on this machine with `npm install -g /app/csvsum` (the machine is offline).
EOF
cat > /data/sales.csv <<'EOF'
order_id,region,amount,discount
1001,north,120.00,5
1002,south,75.50,
1003,north,19.99,2
1004,east,,
1005,west,310.25,15
1006,south,42.00,0
EOF
