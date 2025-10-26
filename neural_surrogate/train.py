import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_SIZE, OUTPUT_SIZE, GHOST_CELLS, GRADIENT_LOSS_WEIGHT, UNROLL_STEPS
from model import pad_with_ghost_cells

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_criterion = nn.MSELoss()

# --- Configuration for the loss function ---
domain_length = 1 # length of the subdomain, only for dx calculation
dx = domain_length / OUTPUT_SIZE
loss_kwargs = {
	'base_criterion': base_criterion, 
	'internal_weight': 1.0,
	'gradient_weight': GRADIENT_LOSS_WEIGHT, # weight for the gradient loss
	'dx': dx,
	'ghost_cells': GHOST_CELLS
}

def loss_w_gradient(pred_padded, target_padded, base_criterion,
                    internal_weight=1.0, gradient_weight=0.1, dx=1.0, ghost_cells=GHOST_CELLS, second_gradient=True):
    
    # detatch grad for target
    target_padded = target_padded.detach()

    # 1. Internal Field Loss
    pred_internal = pred_padded[:, ghost_cells//2:-ghost_cells//2]
    target_internal = target_padded[:, ghost_cells//2:-ghost_cells//2]
    
    # 2. Sobolev Space Gradient Loss
    # Compute spatial gradients using first order forward difference
    grad_pred = (pred_padded[:, 1:] - pred_padded[:, :-1]) / dx
    grad_target = (target_padded[:, 1:] - target_padded[:, :-1]) / dx

    min_u_true = target_internal.min(dim=1, keepdim=True)[0]
    max_u_true = target_internal.max(dim=1, keepdim=True)[0]
    range_u_true = max_u_true - min_u_true

    min_grad_true = grad_target.min(dim=1, keepdim=True)[0]
    max_grad_true = grad_target.max(dim=1, keepdim=True)[0]
    range_grad_true = max_grad_true - min_grad_true

    # avoid division by zero
    epsilon = 1e-6
    range_u_true = torch.clamp(range_u_true, min=epsilon)
    range_grad_true = torch.clamp(range_grad_true, min=epsilon)

    pred_internal_scaled = (pred_internal - min_u_true) / range_u_true
    target_internal_scaled = (target_internal - min_u_true) / range_u_true

    grad_pred_scaled = (grad_pred - min_grad_true) / range_grad_true
    grad_target_scaled = (grad_target - min_grad_true) / range_grad_true

    loss_internal = base_criterion(pred_internal_scaled, target_internal_scaled)
    loss_gradient = base_criterion(grad_pred_scaled, grad_target_scaled)
    if loss_gradient > loss_internal:
        loss_gradient = torch.clamp(loss_gradient, max=loss_internal.detach())

    # print(f"Internal Loss: {loss_internal.item() * internal_weight:.3e}")
    # print(f"Gradient Loss: {loss_gradient.item() * gradient_weight:.3e}")

    # 3. Second Gradient Loss (optional)

    if second_gradient:
        # Take gradient of gradient
        grad2_pred = (grad_pred[:, 2:] - grad_pred[:, :-2]) / (2*dx)
        grad2_target = (grad_target[:, 2:] - grad_target[:, :-2]) / (2*dx)
        # scale loss gradient to match loss_internal
        min_grad2_true = grad2_target.min(dim=1, keepdim=True)[0]
        max_grad2_true = grad2_target.max(dim=1, keepdim=True)[0]
        range_grad2_true = max_grad2_true - min_grad2_true
        range_grad2_true = torch.clamp(range_grad2_true, min=epsilon)

        grad2_pred_scaled = (grad2_pred - min_grad2_true) / range_grad2_true
        grad2_target_scaled = (grad2_target - min_grad2_true) / range_grad2_true

        loss_gradient_2 = base_criterion(grad2_pred_scaled, grad2_target_scaled) # second gradient is more unstable
        if loss_gradient_2 > loss_internal:
            loss_gradient_2 = torch.clamp(loss_gradient_2, max=loss_internal.detach())


        # print(f"2nd Gradient Loss: {loss_gradient_2.item() * gradient_weight:.3e}")
        loss_gradient += loss_gradient_2 / 10
    
    # Total Weighted Loss
    total_loss = (internal_weight * loss_internal +
                  gradient_weight * loss_gradient)
    
    return total_loss

def unroll_loss(x_padded, y_padded, model, criterion=base_criterion, unroll_steps=UNROLL_STEPS, gamma=0.9, eval=False, gradient_weight=GRADIENT_LOSS_WEIGHT):
	if unroll_steps == -1:
		unroll_steps = y_padded.shape[1]

	loss = torch.tensor(0.0, device=device)
	prev_step_unpadded = x_padded[:, GHOST_CELLS//2:-GHOST_CELLS//2]

	if eval:
		gamma = 1.0 # no attenuation during evaluation
		# loss_kwargs['gradient_weight'] = 0.0 # no gradient loss during evaluation
	# ---

	bc_left_t = x_padded[:, :GHOST_CELLS//2]
	bc_right_t = x_padded[:, -GHOST_CELLS//2:]

	# --- Loss Attenuation ---
	total_weight = 0.0 # To normalize the final loss

	for i in range(unroll_steps):

		bc_left_t_next = y_padded[:, i, :GHOST_CELLS//2]
		bc_right_t_next = y_padded[:, i, -GHOST_CELLS//2:]

		padded_input = pad_with_ghost_cells(prev_step_unpadded, bc_left_t, bc_right_t)
		pred_unpadded = model(padded_input)

		# assert padded_input.shape[1] == INPUT_SIZE, f"Model input size {padded_input.shape[1]} does not match expected {INPUT_SIZE}"
		# assert pred_unpadded.shape[1] == OUTPUT_SIZE, f"Model output size {pred_unpadded.shape[1]} does not match expected {OUTPUT_SIZE}"
		
		# Assemble the 130-point tensors for the loss function
		pred_padded = pad_with_ghost_cells(pred_unpadded, bc_left_t_next, bc_right_t_next)
		target_padded = y_padded[:, i]
		
		if criterion == loss_w_gradient:
			step_loss = criterion(pred_padded, target_padded, **loss_kwargs)
		else:
			step_loss = criterion(pred_padded, target_padded)

		# --- Loss Attenuation ---
		weight = gamma ** i
		loss += weight * step_loss
		total_weight += weight
		
		prev_step_unpadded = pred_unpadded

		bc_left_t = bc_left_t_next
		bc_right_t = bc_right_t_next
		
	return loss / total_weight

def train_bptt(criterion, optimizer, scheduler, model, train_loader, val_loader, device, UNROLL_STEPS, EPOCHS, EARLY_STOPPING, NUM_STEPS_PER_EPOCH, model_name="model"):
	unroll_steps = UNROLL_STEPS
	early_stopping = EARLY_STOPPING

	best_epoch = 0
	best_val_loss = float("inf")
	current_lr = optimizer.param_groups[0]['lr']

	loss_history = {"train": [], "val": []}
	print("Starting training...")

	for epoch in range(EPOCHS):
		train_loss = 0.0
		for (step), (x_padded, y_padded) in enumerate(train_loader):
			model.train() # set model to training mode to enable dropout, batchnorm, etc.
			x_padded = x_padded.to(device)
			y_padded = y_padded.to(device)
			
			optimizer.zero_grad()
			loss = unroll_loss(x_padded, y_padded, model, criterion=criterion, unroll_steps=UNROLL_STEPS, gamma=0.9, eval=False, gradient_weight=GRADIENT_LOSS_WEIGHT)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
			if step >= NUM_STEPS_PER_EPOCH - 1:
				break

		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for (step), (x_padded, y_padded) in enumerate(val_loader):
				x_padded = x_padded.to(device)
				y_padded = y_padded.to(device)
				loss = unroll_loss(x_padded, y_padded, model, criterion=criterion, unroll_steps=UNROLL_STEPS, gamma=1, eval=True, gradient_weight=GRADIENT_LOSS_WEIGHT)
				val_loss += loss.item()
				if step >= NUM_STEPS_PER_EPOCH - 1:
					break

		avg_train = train_loss / NUM_STEPS_PER_EPOCH
		avg_val = val_loss / NUM_STEPS_PER_EPOCH

		if avg_val < best_val_loss:
			best_val_loss = avg_val
			best_epoch = epoch
			best_lr = current_lr 
			best_model_wts = {k: v.cpu() for k, v in model.state_dict().items()}
			print(f"New best model found at epoch {epoch+1} with val loss {best_val_loss:.4e}")
			torch.save(best_model_wts, f"models/best_{model_name}.pth")

		# Get the current learning rate from the optimizer
		current_lr = optimizer.param_groups[0]['lr']
		
		print(
			f"Epoch {epoch+1}, "
			f"Train Loss: {avg_train:.4e}, "
			f"Val Loss: {avg_val:.4e}, "
			f"LR: {current_lr:.2e}"
		)

		loss_history["train"].append(avg_train)
		loss_history["val"].append(avg_val)

		# Step the scheduler at the end of each epoch
		scheduler.step()

		if epoch - best_epoch > early_stopping:
			print(f"Early stopping triggered. Stopping training at epoch {epoch+1}.")
			break

	# Load best model weights
	model.load_state_dict(best_model_wts)

	print("Finished training.")

	loss_history["best_lr"] = best_lr

	return loss_history