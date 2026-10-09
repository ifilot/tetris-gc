// Differential oracle against the original MIT-licensed JavaScript implementation.
const fs = require('fs'), vm = require('vm'), path = require('path');
for (const file of ['piece.js', 'grid.js', 'ai.js'])
    vm.runInThisContext(fs.readFileSync(path.join(__dirname, file), 'utf8'));
const ids = [6, 0, 5, 4, 3, 1, 2];
const input = fs.readFileSync(0, 'utf8').trim();
if (input) for (const line of input.split('\n')) {
    const v = line.trim().split(/\s+/).map(Number);
    const grid = new Grid(22, 10);
    for (let y=0;y<22;++y) for(let x=0;x<10;++x) grid.cells[21-y][x] = (v[6+y] >> x) & 1;
    const ai = new AI({heightWeight:v[2], holesWeight:v[3], bumpinessWeight:v[4], linesWeight:-v[5]});
    const pieces = [Piece.fromIndex(ids[v[0]])];
    if(v[1]>=0) pieces.push(Piece.fromIndex(ids[v[1]]));
    const chosen = ai.best(grid, pieces);
    if(chosen === null) console.log('null');
    else {
        while(chosen.moveDown(grid));
        grid.addPiece(chosen);
        console.log(Array.from({length:22},(_,y)=>grid.cells[21-y].reduce((mask,c,x)=>mask|(c?1<<x:0),0)).join(' '));
    }
}
