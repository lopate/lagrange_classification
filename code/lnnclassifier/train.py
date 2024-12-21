import jax
import jax.numpy as jnp
from jax.example_libraries import stax
from jax.example_libraries import optimizers
from jax.experimental.ode import odeint
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

TRAIN_DATASET_PATH = Path("trajectories")
TEST_DATASET_PATH = Path("trajectories")
LOG_DIR = Path("logs")

# neural network model
init_random_params, nn_forward_fn = stax.serial(
        stax.Dense(128),
        stax.Softplus,
        stax.Dense(128),
        stax.Softplus,
        stax.Dense(1),
    )

def equation_of_motion(lagrangian, state, t=None):
    """
    getting acceleration
    """
    q, q_t = jnp.split(state, 2)
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
            @ (jax.grad(lagrangian, 0)(q, q_t)
                - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
    return jnp.concatenate(q_tt)

def normalize_dp(state):
    """
    wrap generalized coordinates to [-pi, pi]
    """
    return jnp.concatenate([(state[:2] + np.pi) % (2 * np.pi) - np.pi, state[2:]])

def learned_lagrangian(params):
    """
    replace the lagrangian with a parameteric model
    """
    def lagrangian(q, q_t):
        assert q.shape == (3,)
        state = normalize_dp(jnp.concatenate([q, q_t]))
        return jnp.squeeze(nn_forward_fn(params, state), axis=-1)
    return lagrangian

@jax.jit
def loss(params, batch):
    """
    define the loss of the model
    """
    state, targets = batch
    preds = jax.vmap(partial(equation_of_motion, learned_lagrangian(params)))(state)
    return jnp.mean((preds - targets) ** 2)

def plot_loss(train_losses, test_losses, log_dir):
    """
    plot of loss
    """
    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.title('Losses over training')
    plt.xlabel("Train step")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(log_dir, "loss.png"), dpi=150)

def train(init_random_params, x_train, xt_train, x_test, xt_test, log_dir=None, name=""):
    """
    model training
    """
    rng = jax.random.PRNGKey(0)
    _, init_params = init_random_params(rng, (-1, 6))

    batch_size = 100
    test_every = 10
    num_batches = 100

    train_losses = []
    test_losses = []

    # adam w learn rate decay
    opt_init, opt_update, get_params = optimizers.adam(
        lambda t: jnp.select([t < batch_size * (num_batches // 3),
                            t < batch_size * (2 * num_batches // 3),
                            t > batch_size * (2 * num_batches // 3)],
                            [1e-3, 3e-4, 1e-4]))
    opt_state = opt_init(init_params)

    @jax.jit
    def update_derivative(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, jax.grad(loss)(params, batch), opt_state)

    for iteration in range(batch_size * num_batches + 1):
        if iteration % batch_size == 0:
            params = get_params(opt_state)
            train_loss = loss(params, (x_train, xt_train))
            train_losses.append(train_loss)
            test_loss = loss(params, (x_test, xt_test))
            test_losses.append(test_loss)
            if iteration % (batch_size*test_every) == 0:
                print(f"iteration={iteration}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")
        opt_state = update_derivative(iteration, opt_state, (x_train, xt_train))

    params = get_params(opt_state)

    params_arr = [np.hstack([np.array(param_subgroup).reshape(-1) for param_subgroup in param_group]) \
                  if len(param_group) > 0 else np.array([]) for param_group in params]

    params_numpy = np.hstack(params_arr)
    
    if log_dir is not None:
        with open(Path(log_dir, f"{name}"), 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        plot_loss(train_losses, test_losses, log_dir)

def main():

    experiment_name = str(sys.argv[1])
    n = int(sys.argv[2])
    start_idx = int(sys.argv[3])
    jax_gpu= jax.devices("gpu")
    print(jax_gpu)
    print(jax.default_backend())
    for i in range(start_idx, start_idx + n):
        name = experiment_name + f'_{i + 1}.pickle'
        
        path = Path(TRAIN_DATASET_PATH, name)

        with open(path, 'rb') as f:
            train_data = pickle.load(f)
            
        with open(path, 'rb') as f:
            test_data = pickle.load(f)
    
        [x_train, xt_train] = train_data
        [x_test, xt_test] = test_data

        x_train = jax.device_put(jax.vmap(normalize_dp)(x_train))
        x_test = jax.device_put(jax.vmap(normalize_dp)(x_test))
        
        print(f"======== {name} ========")
        train(init_random_params, x_train, xt_train, x_test, xt_test, LOG_DIR, name = name)
        
    print("======== done! =========")

if __name__ == '__main__':
    main()
