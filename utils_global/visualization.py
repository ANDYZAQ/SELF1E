import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
from utils_internvl.utils import IMG_CONTEXT_TOKEN, IMG_END_TOKEN

def wrap_text(text, font, font_scale, max_width):
    """
    auto wrap text, ensure each line does not exceed the specified width
    """
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        text_size = cv2.getTextSize(test_line, font, font_scale, 1)[0]
        
        if text_size[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                # if a word is too long, force wrap
                lines.append(word)
    
    if current_line:
        lines.append(current_line)
    
    return lines

def visualize_mask(image_path, mask, gt_mask, conversation, path, idx, iou, dataset_name):
    # Read and convert image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert masks to numpy arrays
    mask = mask.cpu().numpy() # (H, W)
    gt_mask = gt_mask.cpu().numpy() # (H, W)
    # print(f"mask: {mask.shape}, gt_mask: {gt_mask.shape}")

    # Resize image, mask, gt_mask to 200x200
    image = cv2.resize(image, (200, 200)) # (200, 200, 3)
    mask = cv2.resize(mask.astype(np.float32), (200, 200)) # (200, 200)
    gt_mask = cv2.resize(gt_mask.astype(np.float32), (200, 200)) # (200, 200)
    
    # Create visualization for predicted mask (red)
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb_image[mask == 0] = [0, 0, 0]
    rgb_image[mask > 0] = [255, 0, 0]
    pred_vis = cv2.addWeighted(image, 0.5, rgb_image, 0.5, 0)
    
    # Create visualization for ground truth mask (green) 
    rgb_image_gt = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    rgb_image_gt[gt_mask == 0] = [0, 0, 0]
    rgb_image_gt[gt_mask > 0] = [0, 255, 0]
    gt_vis = cv2.addWeighted(image, 0.5, rgb_image_gt, 0.5, 0)

    # Concatenate the visualizations horizontally
    combined_vis = np.concatenate((pred_vis, gt_vis), axis=1)

    # text processing parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    line_spacing = 25  # line spacing
    margin = 10  # margin
    
    # calculate the available text area width
    max_text_width = combined_vis.shape[1] - 2 * margin
    
    # process conversation text
    conversation_text = conversation.replace("\n", " ")
    
    # auto wrap text
    text_lines = wrap_text(conversation_text, font, font_scale, max_text_width)
    
    # calculate the required blank area height
    text_height = len(text_lines) * line_spacing + 2 * margin
    blank_height = max(100, text_height)  # at least 100 pixels high
    
    # add blank area
    combined_vis = np.concatenate((combined_vis, np.zeros((blank_height, combined_vis.shape[1], 3), dtype=np.uint8)), axis=0)
    
    # draw text
    start_y = combined_vis.shape[0] - blank_height + margin + line_spacing
    for i, line in enumerate(text_lines):
        y_pos = start_y + i * line_spacing
        # ensure text does not exceed the bottom boundary
        if y_pos < combined_vis.shape[0] - margin:
            cv2.putText(combined_vis, line, (margin, y_pos), font, font_scale, (255, 255, 255), font_thickness)
    
    # save the combined visualization
    os.makedirs(os.path.join(path, dataset_name), exist_ok=True)
    combined_vis_bgr = cv2.cvtColor(combined_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(path, dataset_name, f'{idx}_{iou:.2f}.png'), combined_vis_bgr)


def visualize_entropy(soft_mask, path, idx, iou, dataset_name):
    seg_prob_sigmoid = torch.sigmoid(soft_mask)
    seg_prob_entropy_pixel = -(seg_prob_sigmoid * torch.log(seg_prob_sigmoid + 1e-10))
    seg_prob_entropy_pixel = seg_prob_entropy_pixel.float().cpu().numpy()
    assert len(seg_prob_entropy_pixel.shape) == 2, f"seg_prob_entropy_pixel.shape: {seg_prob_entropy_pixel.shape}"
    
    # Calculate target size keeping aspect ratio with max side 200
    h, w = seg_prob_entropy_pixel.shape
    scale = 200.0 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Create figure with correct size in inches (dpi=100 by default)
    plt.figure(figsize=(new_w/100, new_h/100))
    
    # Plot with proper aspect ratio
    plt.imshow(seg_prob_entropy_pixel, cmap='plasma', interpolation='nearest')
    plt.colorbar()
    
    # Remove axes and padding
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    # Save and cleanup
    os.makedirs(os.path.join(path, dataset_name), exist_ok=True)
    plt.savefig(os.path.join(path, dataset_name, f'{idx}_{iou:.2f}_entropy.png'), 
                bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()
    plt.clf()


def visualize_attention(attention, conversation, path, idx, iou):
    """
    efficient visualization of attention matrix
    Args:
        attention: attention matrix (seq_len, seq_len) or (num_heads, seq_len, seq_len)
        conversation: token list
        path: save path
        idx: file index
        iou: IoU value
    """
    
    # if 3D tensor (num_heads, seq_len, seq_len), average all heads
    if attention.ndim == 3:
        attention = attention.mean(dim=0)
    # ensure attention is a numpy array
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().float().cpu().numpy()
    
    # process -inf values, set a very small value for visualization
    attention_vis = np.copy(attention)
    mask_neginf = np.isinf(attention_vis) & (attention_vis < 0)
    attention_vis[mask_neginf] = 0
    
    # quickly process tokens
    seq_len = attention.shape[0]
    if len(conversation) > seq_len:
        tokens = conversation[:seq_len]
    elif len(conversation) < seq_len:
        tokens = conversation + [f"<pad_{i}>" for i in range(seq_len - len(conversation))]
    else:
        tokens = conversation
    
    # simplify token display (only truncate, no complex judgment)
    display_tokens = [str(token)[:6] + "..." if len(str(token)) > 6 else str(token) for token in tokens]
    
    # use smaller image size
    fig_size = seq_len * 0.15
    plt.figure(figsize=(fig_size, fig_size))
    
    # improve heatmap configuration - enhance color distinction
    # calculate the effective attention value range (exclude 0 values)
    nonzero_attention = attention_vis[attention_vis > 0]
    if len(nonzero_attention) > 0:
        vmin = np.percentile(nonzero_attention, 5)   # use 5% percentile as the minimum value
        vmax = np.percentile(nonzero_attention, 95)  # use 95% percentile as the maximum value
        
        # if the range is too small, expand the range to enhance contrast
        if vmax - vmin < 0.01:
            mean_val = np.mean(nonzero_attention)
            vmin = max(0, mean_val - 0.02)
            vmax = mean_val + 0.02
    else:
        vmin, vmax = 0, 1
    
    # slightly enhance contrast of attention values
    attention_enhanced = np.power(attention_vis, 0.8)  # use power function to enhance contrast
    attention_enhanced[mask_neginf] = 0  # keep mask area as 0
    
    # use high contrast color mapping
    plt.imshow(attention_enhanced, cmap='plasma', aspect='auto', interpolation='nearest', 
               vmin=vmin**0.8, vmax=vmax**0.8)
    
    # add high contrast color bar
    cbar = plt.colorbar(label='Attention Weight', shrink=0.8, format='%.4f')
    cbar.ax.tick_params(labelsize=8)
    
    # set ticks (full display)
    plt.xticks(range(seq_len), display_tokens, rotation=60, ha='right', fontsize=8)
    plt.yticks(range(seq_len), display_tokens, rotation=60, fontsize=8)
    
    # adjust layout
    plt.tight_layout()
    
    # save image
    os.makedirs(path, exist_ok=True)
    attention_path = os.path.join(path, f'{idx}_{iou:.2f}_attention.png')
    plt.savefig(attention_path, dpi=72, bbox_inches='tight', facecolor='white')
    plt.close()
    plt.clf()  # clear image cache

def visualize_img_cont_attention(attention, conversation, path, idx, iou, dataset_name):
    """
    get attention between image content and conversation content, visualize all image heatmaps
    """
    # if 3D tensor (num_heads, seq_len, seq_len), average all heads
    if attention.ndim == 3:
        attention = attention.mean(dim=0)
    # ensure attention is a numpy array
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().float().cpu().numpy()

    seq_len = attention.shape[0]
    if len(conversation) > seq_len:
        conversation = conversation[:seq_len]
    elif len(conversation) < seq_len:
        conversation = conversation + [f"<pad_{i}>" for i in range(seq_len - len(conversation))]
    else:
        conversation = conversation

    # get attention between image content and conversation content
    img_cont_pos = [conv==IMG_CONTEXT_TOKEN for conv in conversation]
    img_cont_pos = np.array(img_cont_pos)
    # img_cont_pos = np.concatenate([img_cont_pos[1:], np.zeros_like(img_cont_pos[:1])], axis=0)
    img_end_pos = [conv==IMG_END_TOKEN for conv in conversation]
    img_end_pos = np.array(img_end_pos).nonzero()[-1][0]
    
    attn_cont = attention[img_end_pos:]
    attn_cont_img = attn_cont[:, img_cont_pos]

    h, w = int(np.sqrt(attn_cont_img.shape[-1])), int(np.sqrt(attn_cont_img.shape[-1]))
    attn_cont_img = attn_cont_img.reshape(-1, h, w)

    # build multiple image layouts based on quantity, show attn evolution process
    n_img = attn_cont_img.shape[0]
    n_row = int(np.ceil(np.sqrt(n_img)))
    n_col = int(np.ceil(n_img / n_row))

    fig, axs = plt.subplots(n_row, n_col, figsize=(10, 10))
    for i in range(n_img):
        axs[i // n_col, i % n_col].imshow(attn_cont_img[i], cmap='plasma', aspect='auto', interpolation='nearest')
        axs[i // n_col, i % n_col].set_title(f'{conversation[img_end_pos+i]}')
        axs[i // n_col, i % n_col].axis('off')

    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, dataset_name, f'{idx}_{iou:.2f}_img_cont_attention-1.png'), dpi=72, bbox_inches='tight', facecolor='white')
    plt.close()
    plt.clf()  # clear image cache
    
def visualize_img_cont_attention_var(attention, conversation, path, idx, iou, dataset_name):
    """
    get attention between image content and conversation content, visualize all image variance heatmaps
    """
    # if 3D tensor (num_heads, seq_len, seq_len), average all heads
    if attention.ndim == 3:
        # Calculate mean for each position across heads
        attention = attention.var(dim=0)  # [seq_len, seq_len]
    # ensure attention is a numpy array
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().float().cpu().numpy()

    seq_len = attention.shape[0]
    if len(conversation) > seq_len:
        conversation = conversation[:seq_len]
    elif len(conversation) < seq_len:
        conversation = conversation + [f"<pad_{i}>" for i in range(seq_len - len(conversation))]
    else:
        conversation = conversation
        
    img_cont_pos = [conv==IMG_CONTEXT_TOKEN for conv in conversation]
    img_cont_pos = np.array(img_cont_pos)
    img_end_pos = [conv==IMG_END_TOKEN for conv in conversation]
    img_end_pos = np.array(img_end_pos).nonzero()[-1][0]
    
    attn_cont = attention[img_end_pos:]
    attn_cont_img = attn_cont[:, img_cont_pos]

    h, w = int(np.sqrt(attn_cont_img.shape[-1])), int(np.sqrt(attn_cont_img.shape[-1]))
    attn_cont_img = attn_cont_img.reshape(-1, h, w)

    # build multiple image layouts based on quantity, show attn evolution process
    n_img = attn_cont_img.shape[0]
    n_row = int(np.ceil(np.sqrt(n_img)))
    n_col = int(np.ceil(n_img / n_row))
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(10, 10))
    for i in range(n_img):
        axs[i // n_col, i % n_col].imshow(attn_cont_img[i], cmap='plasma', aspect='auto', interpolation='nearest')
        axs[i // n_col, i % n_col].set_title(f'{conversation[img_end_pos+i]}')
        axs[i // n_col, i % n_col].axis('off')

    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, dataset_name, f'{idx}_{iou:.2f}_img_cont_attention_var.png'), dpi=72, bbox_inches='tight', facecolor='white')
    plt.close()
    plt.clf()  # clear image cache