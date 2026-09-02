[checkpoint](thorcino/optimizer.py)
[checkpoint](thorcino/training/schedulers.py)

4. Optimizer e scheduler non vengono mai ripristinati
save() serializza optimizer_state e scheduler_state, ma load() li ha commentati (trainer.py:200-205) e i metodi _set_optimizer_state/_set_scheduler_state non esistono — decommentare così com'è dà AttributeError. Conseguenze concrete sul resume:

[checkpoint](thorcino/layers/)

6. Nessuna validazione di shape in set_state
self.weights.data = state['W'] accetta qualsiasi shape: caricare un checkpoint di un'architettura diversa non fallisce, ridimensiona i pesi e rompe il modello più avanti in modo opaco. Un assert self.weights.data.shape == state['W'].shape per parametro rende l'errore immediato.

step_count riparte da 0 → la bias correction di Adam/AdamW (1 - beta1**step_count) riparte da ~0.1, quindi il primo step dopo il resume applica un update ~10× più grande del dovuto.
m_buffers/v_buffers e momentum_buffer si azzerano → si perde tutto lo stato di ottimizzazione.
optimizer.lr non viene ripristinato: con scheduler si autocorregge a fine epoca, ma la prima epoca dopo il resume gira al learning rate iniziale.
Nota anche che step_count non è nemmeno dentro state di nessun optimizer, quindi va aggiunto prima di poterlo ripristinare. E Schedule.set_state esiste (schedulers.py:39) ma non è chiamata da nessuna parte.

Il docstring di load() ("model, optimizer and scheduler state are not restored") è ora sbagliato per il modello: quello viene ripristinato.

[loader](core/loader.py)
1) add parallel loading via multi threading
2) add pre fetching of N+1 batches

[autograd](core/autograd/arithmetic.py)
[autograd](core/autograd/activations.py)
[autograd](core/autograd/losses.py)
2) Add Cache: instend of recomputing base functions, cache it from forward pass and let backward pass reuse it
3) Add Debug step when backward fails: show on which node the failure occurred