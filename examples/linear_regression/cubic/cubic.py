from thorcino.dataset import DataLoader, TensorDataset
from thorcino.optimizer import SGD
from thorcino.tensor import Tensor
from thorcino.losses import MSELoss
from thorcino.training import Trainer, CosineSchedule
from thorcino.layers import Sequential, Linear
import numpy as np
import matplotlib.pyplot as plt

# Configuration
EPOCHS = 75
EVAL_STEP = 5
DATASET_SIZE = 100
NOISY = 120
MAX_LR = 1e-4
MIN_LR = 1e-7


# To estimate function
def f(x: np.ndarray):
    """The function we want to estimate"""
    return 1.2*x**3 - 2.3*x**2 + x*2 + 2


# Dataset
train_domain = np.linspace(-5, 5, DATASET_SIZE).reshape(-1, 1)
test_domain = np.linspace(-10, 10, DATASET_SIZE).reshape(-1, 1)
f_train = f(train_domain)
f_test = f(test_domain)
train_x = Tensor(np.hstack([train_domain, train_domain**2, train_domain**3]))
test_x = Tensor(np.hstack([test_domain, test_domain**2, test_domain**3]))
train_y = Tensor(f_train + np.random.uniform(-1, 1, f_train.shape)*NOISY)
test_y = Tensor(f_test + np.random.uniform(-1, 1, f_test.shape)*NOISY)

train_dataset = TensorDataset(train_x, train_y)
train_dataloader = DataLoader(train_dataset, DATASET_SIZE)

test_dataset = TensorDataset(test_x, test_y)
test_dataloader = DataLoader(test_dataset, DATASET_SIZE)

# Model architecture
model = Sequential(
    Linear(3, 1),
)
loss = MSELoss()
optimizer = SGD(model.parameters, MAX_LR)
scheduler = CosineSchedule(MAX_LR, MIN_LR, EPOCHS)
trainer = Trainer(
    model,
    loss,
    optimizer,
    scheduler
)

# Save graph of models
#model.save_graph("./examples/linear_regression/cubic/arch/architecture.png", arch=True)
#model.save_graph("./examples/linear_regression/cubic/arch/forward.png", arch=False, forward=True)
#model.save_graph("./examples/linear_regression/cubic/arch/backward.png", arch=False, backward=True)


# Train loop
for i, epoch in enumerate(range(EPOCHS)):
    _ = trainer.train_epoch(train_dataloader, 1)

    if i % EVAL_STEP == 0 :
        _ = trainer.eval(test_dataloader)


# Eval
out = model(test_x)


# Plot in one window, divided into sections
fig, (ax_loss, ax_result) = plt.subplots(1, 2, figsize=(12, 5))

# Loss section
ax_loss.plot(list(range(EPOCHS)), trainer.train_loss, label="train")
ax_loss.plot(list(range(0, EPOCHS, EVAL_STEP)), trainer.eval_loss, label="test")
ax_loss.set_title("Loss")
ax_loss.set_xlabel("epoch")
ax_loss.set_ylabel("loss")
ax_loss.legend()


# Result section (the regression)
ax_result.plot(test_domain, test_y.data, 'r.', label="target")
ax_result.plot(test_domain, out.data, label="prediction")
ax_result.plot(test_domain, f_test, label='to estimate')
ax_result.set_title("Regression")
ax_result.set_xlabel("x")
ax_result.set_ylabel("y")
ax_result.legend()

fig.tight_layout()
plt.show()
