# ============================================================================
# STEP 1 FIX: Load Preprocessed Data with u_idx Mapping
# ============================================================================

print("="*80)
print("STEP 1: Loading Preprocessed Data from Task 01")
print("="*80)

step1_start = time.time()

# Load training matrix
matrix_path = os.path.join(DATA_DIR, 'X_train_confidence.npz')
if not os.path.exists(matrix_path):
    raise FileNotFoundError(
        f"Training matrix not found: {matrix_path}\n"
        f"Please run Task 01 data pipeline first!"
    )

X_train = load_npz(matrix_path)
print(f"✅ Loaded training matrix: {X_train.shape}")
print(f"   Density: {X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.6f}")
print(f"   Non-zeros: {X_train.nnz:,}")

# Load ID mappings
mappings_path = os.path.join(DATA_DIR, 'user_item_mappings.json')
with open(mappings_path, 'r') as f:
    mappings_data = json.load(f)

# NOTE: user_to_idx maps user_id (string) to u_idx (int)
user_to_idx = mappings_data['user_to_idx']
idx_to_user = {int(k): v for k, v in mappings_data['idx_to_user'].items()}
item_to_idx = mappings_data['item_to_idx']
idx_to_item = {int(k): v for k, v in mappings_data['idx_to_item'].items()}

print(f"✅ Loaded ID mappings:")
print(f"   Users: {len(user_to_idx):,} (trainable users only)")
print(f"   Items: {len(item_to_idx):,}")

# Extract metadata
metadata = {
    'positive_threshold': mappings_data.get('positive_threshold', 4.0),
    'hard_negative_threshold': mappings_data.get('hard_negative_threshold', 3.0),
    'data_hash': mappings_data.get('data_hash', 'UNKNOWN_HASH'),
    'timestamp': mappings_data.get('timestamp')
}

data_hash_display = metadata['data_hash'][:8] if metadata['data_hash'] else "N/A"
print(f"✅ Metadata: positive_threshold={metadata['positive_threshold']}, data_hash={data_hash_display}...")

# Load user sets with u_idx mapping
# NOTE: These files have already been converted to use u_idx instead of user_id
train_sets_path = os.path.join(DATA_DIR, 'user_pos_train_u_idx.pkl')
test_sets_path = os.path.join(DATA_DIR, 'user_pos_test_u_idx.pkl')

# Fallback to original files if converted versions don't exist
if not os.path.exists(train_sets_path) or not os.path.exists(test_sets_path):
    print("⚠️  Converted u_idx files not found, attempting conversion...")
    
    # Load original files
    train_sets_path_orig = os.path.join(DATA_DIR, 'user_pos_train.pkl')
    test_sets_path_orig = os.path.join(DATA_DIR, 'user_pos_test.pkl')
    
    with open(train_sets_path_orig, 'rb') as f:
        user_pos_train_orig = pickle.load(f)
    with open(test_sets_path_orig, 'rb') as f:
        user_pos_test_orig = pickle.load(f)
    
    # Convert user_id to u_idx
    user_pos_train = {}
    user_pos_test = {}
    
    for user_id, items in user_pos_train_orig.items():
        user_id_str = str(user_id)
        if user_id_str in user_to_idx:
            u_idx = user_to_idx[user_id_str]
            user_pos_train[u_idx] = items
    
    for user_id, items in user_pos_test_orig.items():
        user_id_str = str(user_id)
        if user_id_str in user_to_idx:
            u_idx = user_to_idx[user_id_str]
            user_pos_test[u_idx] = items
    
    print(f"   Converted {len(user_pos_train)} train users to u_idx")
    print(f"   Converted {len(user_pos_test)} test users to u_idx")
else:
    # Load pre-converted files
    with open(train_sets_path, 'rb') as f:
        user_pos_train = pickle.load(f)
    with open(test_sets_path, 'rb') as f:
        user_pos_test = pickle.load(f)

print(f"✅ Loaded user sets (with u_idx):")
print(f"   Train users: {len(user_pos_train):,}")
print(f"   Test users: {len(user_pos_test):,}")

# Load data stats (optional)
stats_path = os.path.join(DATA_DIR, 'data_stats.json')
data_stats = None
if os.path.exists(stats_path):
    with open(stats_path, 'r') as f:
        data_stats = json.load(f)
    print(f"✅ Loaded data statistics")
else:
    print(f"⚠️  Data stats not found (optional)")

# Extract validation users
np.random.seed(CONFIG['random_seed'])
test_users = list(user_pos_test.keys())
num_val = max(1, int(len(test_users) * CONFIG['val_ratio']))
val_users = np.random.choice(test_users, num_val, replace=False).tolist()
remaining_test_users = list(set(test_users) - set(val_users))

print(f"✅ Extracted validation users:")
print(f"   Validation: {len(val_users):,} ({CONFIG['val_ratio']*100:.1f}%)")
print(f"   Test: {len(remaining_test_users):,}")

step1_time = time.time() - step1_start
print(f"\n⏱️  Step 1 completed in {step1_time:.2f}s")
print("="*80)
