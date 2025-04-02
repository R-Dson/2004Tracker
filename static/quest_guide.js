document.addEventListener('DOMContentLoaded', () => {
    const questContainer = document.querySelector('.quest-detail');
    if (!questContainer) return;

    const questId = questContainer.dataset.questId;
    const savedState = JSON.parse(localStorage.getItem(`questProgress_${questId}`)) || {};

    // Initialize all checkboxes
    document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
        const key = getStorageKey(checkbox);
        checkbox.checked = !!savedState[key];
        checkbox.addEventListener('change', function() {
            updateStorage(this);
            handleStepRelationships(this);
        });
    });

    function getStorageKey(checkbox) {
        if (checkbox.classList.contains('step-checkbox')) {
            return `step_${checkbox.dataset.step}`;
        } else if (checkbox.classList.contains('substep-checkbox')) {
            return `step_${checkbox.dataset.parentStep}_sub_${checkbox.dataset.step}`;
        } else {
            return `${checkbox.className.replace('-checkbox', '')}_${checkbox.dataset.type || checkbox.dataset.item || ''}`;
        }
    }

    function updateStorage(checkbox) {
        if (!questId) return;
        const key = getStorageKey(checkbox);
        const state = JSON.parse(localStorage.getItem(`questProgress_${questId}`)) || {};
        state[key] = checkbox.checked;
        localStorage.setItem(`questProgress_${questId}`, JSON.stringify(state));
    }

    function handleStepRelationships(checkbox) {
        // Handle step-substep relationships
        if (checkbox.classList.contains('step-checkbox')) {
            const substeps = checkbox.closest('.step').querySelectorAll('.substep-checkbox');
            substeps.forEach(substep => {
                substep.checked = checkbox.checked;
                updateStorage(substep);
            });
        } else if (checkbox.classList.contains('substep-checkbox')) {
            const step = checkbox.closest('.step');
            const parentCheckbox = step.querySelector('.step-checkbox');
            const allSubsteps = step.querySelectorAll('.substep-checkbox');
            parentCheckbox.checked = Array.from(allSubsteps).every(sub => sub.checked);
            updateStorage(parentCheckbox);
        }
    }
});