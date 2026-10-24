const form = document.querySelector('#list-form');
const items = document.querySelector('#items');
let dirty = false;
function update(markDirty = true) {
  const rows = [...items.children];
  rows.forEach((row, index) => {
    const checkbox = row.querySelector('[type=checkbox]');
    checkbox.value = index;
    checkbox.setAttribute('aria-label', `Picked up item ${index + 1}`);
    row.querySelector('[name=item_name]').setAttribute('aria-label', `Item ${index + 1}`);
    row.querySelector('[name=quantity]').setAttribute('aria-label', `Quantity for item ${index + 1}`);
    row.querySelector('button').setAttribute('aria-label', `Remove item ${index + 1}`);
    row.classList.toggle('checked', checkbox.checked);
  });
  document.querySelector('#progress').textContent = `${rows.filter(r => r.querySelector('[type=checkbox]').checked).length} of ${rows.length} picked up`;
  document.querySelector('#empty-hint').hidden = rows.length > 0;
  document.querySelector('#add-item').disabled = rows.length >= 200;
  if (markDirty) {
    dirty = true;
    document.querySelector('#save-state').textContent = 'Unsaved changes';
  }
}
document.querySelector('#add-item').addEventListener('click', () => {
  items.append(document.querySelector('#item-template').content.cloneNode(true));
  update();
  items.lastElementChild.querySelector('[name=item_name]').focus();
});
items.addEventListener('click', event => {
  const remove = event.target.closest('.remove');
  if (remove) {
    const next = remove.closest('.item-row').nextElementSibling;
    remove.closest('.item-row').remove();
    update();
    (next?.querySelector('[name=item_name]') || document.querySelector('#add-item')).focus();
  }
});
form.addEventListener('input', () => update());
form.addEventListener('submit', () => { dirty = false; });
window.addEventListener('beforeunload', event => {
  if (dirty) { event.preventDefault(); event.returnValue = ''; }
});
const deleteForm = document.querySelector('.delete-form');
if (deleteForm) {
  const dialog = document.querySelector('#delete-dialog');
  deleteForm.addEventListener('submit', event => { event.preventDefault(); dialog.showModal(); });
  document.querySelector('#cancel-delete').addEventListener('click', () => dialog.close());
  document.querySelector('#confirm-delete').addEventListener('click', () => { dirty = false; deleteForm.submit(); });
}
update(false);
