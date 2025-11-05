function selectCell(row, col) {
    document.querySelectorAll('.data-table tr').forEach(tr => tr.classList.remove('selected'));
    var tr = document.getElementById('row-' + row);
    if (tr) tr.classList.add('selected');
    var input = document.getElementById('cell-' + row + '-' + col);
    if (input) input.focus();
}