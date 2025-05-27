sgn_train_x, sgn_train_y, sgn_val_x, sgn_val_y = np.zeros((batch_size, 300, 150)), np.zeros((batch_size, 1)), np.zeros((batch_size, 300, 150)), np.zeros((batch_size, 1))

best_metric = float('inf')
total_epochs = -1
cur_tot_epoch = 0

def train_paired(train_ae = True, train_cross = True, train_discrim = True, train_emb_adv = True, run_eval = True, use_emb_adv = True, use_discrim_adv = True, run_sgn_eval = False, save = True, k=3):
    global best_metric
    global total_epochs
    global cur_tot_epoch
    # Assertions
    # assert train_ae or train_cross or train_discrim or train_emb_adv, "At least one of the training objectives must be True"
    assert not (run_sgn_eval and not run_eval), "If run_sgn_eval is True, then run_eval must be True"

    # Store eval values for validation
    eval_X_known, eval_Y_known_action, eval_Y_known_actor, eval_X_rec, eval_Y_rec_action, eval_Y_rec_actor, eval_X, eval_Y_action, eval_Y_actor, eval_Y_initial_actor = [], [], [], [], [], [], [], [], [], []

    # Losses for printing
    losses = []
    rec_loss, cross_loss, end_effector_loss, smoothing_loss, triplet_loss, latent_consistency_loss, privacy_loss, privacy_loss_adv, privacy_loss_coop, privacy_acc_adv, privacy_acc_coop, priv_training_loss, utility_loss, utility_loss_adv, utility_loss_coop, utility_acc_adv, utility_acc_coop, util_training_loss, discriminator_loss, discriminator_train_losses, discriminator_training_acc, priv_coop_training_loss, priv_training_acc, priv_coop_training_acc, util_coop_training_loss, util_training_acc, util_coop_training_acc = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    # Determine if adversaries need to be trained
    train_emb_this_epoch = True
    if emb_clf_update_per_epoch_paired < 1:
        if cur_tot_epoch % round(1 / emb_clf_update_per_epoch_paired) != 0:
            train_emb_this_epoch = False

    for (x1_pos, x1_rot, x2_pos, x2_rot, y1_pos, y1_rot, y2_pos, y2_rot, actors, actions) in train_dl:
        # Move tensors to the configured device
        x1_pos, x1_rot, x2_pos, x2_rot, y1_pos, y1_rot, y2_pos, y2_rot = x1_pos.float().to(device), x1_rot.float().to(device), x2_pos.float().to(device), x2_rot.float().to(device), y1_pos.float().to(device), y1_rot.float().to(device), y2_pos.float().to(device), y2_rot.float().to(device)

        # Remove rotation data if only using position data
        if only_use_pos:
            x1_rot, x2_rot, y1_rot, y2_rot = x1_pos, x2_pos, y1_pos, y2_pos

        # For 1D convolutions, flatten the data
        if one_dimension_conv:
            x1_pos = x1_pos.view(x1_pos.size(0), T, -1)
            x1_rot = x1_rot.view(x1_rot.size(0), T, -1)
            x2_pos = x2_pos.view(x2_pos.size(0), T, -1)
            x2_rot = x2_rot.view(x2_rot.size(0), T, -1)
            y1_pos = y1_pos.view(y1_pos.size(0), T, -1)
            y1_rot = y1_rot.view(y1_rot.size(0), T, -1)
            y2_pos = y2_pos.view(y2_pos.size(0), T, -1)
            y2_rot = y2_rot.view(y2_rot.size(0), T, -1)


        if train_discrim or train_emb_adv:
            # Train the discriminator
            if train_emb_this_epoch:
                it = 1
                if emb_clf_update_per_epoch_paired > 1: it = emb_clf_update_per_epoch_paired
                for _ in range(int(it)):
                    t_priv_loss, t_priv_coop_loss, t_util_loss, t_util_coop_loss, t_discriminator_loss, t_priv_acc, t_util_acc, t_priv_coop_acc, t_util_coop_acc, t_discriminator_acc  = model.train_adv_paired(x1_pos, x1_rot, x2_pos, x2_rot, y1_pos, y1_rot, y2_pos, y2_rot, actors, actions, train_emb=train_emb_adv, train_discrim=train_discrim)

                # Track the loss
                priv_training_loss.append(t_priv_loss)
                priv_coop_training_loss.append(t_priv_coop_loss)
                priv_training_acc.append(t_priv_acc)
                priv_coop_training_acc.append(t_priv_coop_acc)
                util_training_loss.append(t_util_loss)
                util_coop_training_loss.append(t_util_coop_loss)
                util_training_acc.append(t_util_acc)
                util_coop_training_acc.append(t_util_coop_acc)
                discriminator_train_losses.append(t_discriminator_loss)
                discriminator_training_acc.append(t_discriminator_acc)

        # Zero the gradients
        optimizer.zero_grad()

        # Train the autoencoder/cross reconstruction
        if train_ae or train_cross:
            # Forward pass
            loss, _, _, _, _, losses_ = model.loss_paired(x1_pos, x1_rot, x2_pos, x2_rot, y1_pos, y1_rot, y2_pos, y2_rot, actors, actions, cross=train_cross, reconstruction=train_ae, emb_adv=use_emb_adv, discrim_adv=use_discrim_adv)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Track the loss
            losses.append(loss.item())
            rec_loss.append(losses_['rec_loss'])
            cross_loss.append(losses_['cross_loss'])
            end_effector_loss.append(losses_['end_effector_loss'])
            smoothing_loss.append(losses_['smoothing_loss'])
            latent_consistency_loss.append(losses_['latent_consistency_loss'])
            triplet_loss.append(losses_['triplet_loss'])
            privacy_loss.append(losses_['privacy_loss'])
            privacy_loss_adv.append(losses_['privacy_loss_adv'])
            privacy_loss_coop.append(losses_['privacy_loss_coop'])
            privacy_acc_adv.append(losses_['privacy_acc_adv'])
            privacy_acc_coop.append(losses_['privacy_acc_coop'])
            utility_loss.append(losses_['utility_loss'])
            utility_loss_adv.append(losses_['utility_loss_adv'])
            utility_loss_coop.append(losses_['utility_loss_coop'])
            utility_acc_adv.append(losses_['utility_acc_adv'])
            utility_acc_coop.append(losses_['utility_acc_coop'])
            discriminator_loss.append(losses_['discriminator_loss'])
            discriminator_training_acc.append(losses_['discriminator_acc'])

    # Decay learning rate (disabled for training stages)
    # scheduler.step()

    # Validation
    if run_eval:
        with torch.no_grad():
            val_losses = []
            val_rec_loss, val_cross_loss, val_end_effector_loss, val_smoothing_loss, val_triplet_loss, val_latent_consistency_loss, val_privacy_loss, val_privacy_loss_adv, val_privacy_loss_coop, val_privacy_acc_adv, val_privacy_acc_coop, val_utility_loss, val_utility_loss_adv, val_utility_loss_coop, val_utility_acc_adv, val_utility_acc_coop, val_discriminator_loss, val_discriminator_acc = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

            for (x1_pos, x1_rot, x2_pos, x2_rot, y1_pos, y1_rot, y2_pos, y2_rot, actors, actions) in val_dl:
                x1_pos, x1_rot, x2_pos, x2_rot, y1_pos, y1_rot, y2_pos, y2_rot = x1_pos.float().to(device), x1_rot.float().to(device), x2_pos.float().to(device), x2_rot.float().to(device), y1_pos.float().to(device), y1_rot.float().to(device), y2_pos.float().to(device), y2_rot.float().to(device)

                # Remove rotation data if only using position data
                if only_use_pos:
                    x1_rot, x2_rot, y1_rot, y2_rot = x1_pos, x2_pos, y1_pos, y2_pos

                # For 1D convolutions, flatten the data
                if one_dimension_conv:
                    x1_pos = x1_pos.view(x1_pos.size(0), T, -1)
                    x1_rot = x1_rot.view(x1_rot.size(0), T, -1)
                    x2_pos = x2_pos.view(x2_pos.size(0), T, -1)
                    x2_rot = x2_rot.view(x2_rot.size(0), T, -1)
                    y1_pos = y1_pos.view(y1_pos.size(0), T, -1)
                    y1_rot = y1_rot.view(y1_rot.size(0), T, -1)
                    y2_pos = y2_pos.view(y2_pos.size(0), T, -1)
                    y2_rot = y2_rot.view(y2_rot.size(0), T, -1)

                loss, x1_hat, x2_hat, y1_hat, y2_hat, losses_ = model.loss_paired(x1_pos, x1_rot, x2_pos, x2_rot, y1_pos, y1_rot, y2_pos, y2_rot, actors, actions, cross=train_cross, reconstruction=train_ae, emb_adv=use_emb_adv, discrim_adv=use_discrim_adv)
                val_losses.append(loss.item())
                val_rec_loss.append(losses_['rec_loss'])
                val_cross_loss.append(losses_['cross_loss'])
                val_end_effector_loss.append(losses_['end_effector_loss'])
                val_smoothing_loss.append(losses_['smoothing_loss'])
                val_triplet_loss.append(losses_['triplet_loss'])
                val_latent_consistency_loss.append(losses_['latent_consistency_loss'])
                val_privacy_loss.append(losses_['privacy_loss'])
                val_privacy_loss_adv.append(losses_['privacy_loss_adv'])
                val_privacy_loss_coop.append(losses_['privacy_loss_coop'])
                val_privacy_acc_adv.append(losses_['privacy_acc_adv'])
                val_privacy_acc_coop.append(losses_['privacy_acc_coop'])
                val_utility_loss.append(losses_['utility_loss'])
                val_utility_loss_adv.append(losses_['utility_loss_adv'])
                val_utility_loss_coop.append(losses_['utility_loss_coop'])
                val_utility_acc_adv.append(losses_['utility_acc_adv'])
                val_utility_acc_coop.append(losses_['utility_acc_coop'])
                val_discriminator_loss.append(losses_['discriminator_loss'])
                val_discriminator_acc.append(losses_['discriminator_acc'])

                if run_sgn_eval:
                    if not one_dimension_conv:
                        x1_pos = x1_pos.view(x1_pos.size(0), T, -1)
                        x2_pos = x2_pos.view(x2_pos.size(0), T, -1)
                        y1_pos = y1_pos.view(y1_pos.size(0), T, -1)
                        y2_pos = y2_pos.view(y2_pos.size(0), T, -1)

                    # x1 = P1, A1
                    # x2 = P2, A2
                    # y1 = P1, A2
                    # y2 = P2, A1
                    # actors = A1, A2
                    # actions = P1, P2
                    # x1hat = P1, A1
                    # x2hat = P2, A2
                    # y1hat = P2, A1
                    # y2hat = P1, A2

                    # Raw Data X
                    eval_X_known.append(x1_pos.cpu().numpy())
                    eval_X_known.append(x2_pos.cpu().numpy())
                    eval_X_known.append(y1_pos.cpu().numpy())
                    eval_X_known.append(y2_pos.cpu().numpy())

                    # Raw Data Utility
                    eval_Y_known_action.append(actions[0].cpu().numpy())
                    eval_Y_known_action.append(actions[1].cpu().numpy())
                    eval_Y_known_action.append(actions[1].cpu().numpy())
                    eval_Y_known_action.append(actions[0].cpu().numpy())

                    # Raw Data Privacy
                    eval_Y_known_actor.append(actors[0].cpu().numpy())
                    eval_Y_known_actor.append(actors[1].cpu().numpy())
                    eval_Y_known_actor.append(actors[0].cpu().numpy())
                    eval_Y_known_actor.append(actors[1].cpu().numpy())

                    # Reconstruction X
                    eval_X_rec.append(x1_hat.cpu().numpy())
                    eval_X_rec.append(x2_hat.cpu().numpy())
                    # Cross X
                    eval_X.append(y1_hat.cpu().numpy()) # P2, A1
                    eval_X.append(y2_hat.cpu().numpy()) # P1, A2

                    # Reconstruction Utility
                    eval_Y_rec_action.append(actions[0].cpu().numpy())
                    eval_Y_rec_action.append(actions[1].cpu().numpy())
                    # Cross Utility
                    eval_Y_action.append(actions[0].cpu().numpy())
                    eval_Y_action.append(actions[1].cpu().numpy())

                    # Reconstruction Privacy
                    eval_Y_rec_actor.append(actors[0].cpu().numpy())
                    eval_Y_rec_actor.append(actors[1].cpu().numpy())
                    # Cross Privacy
                    eval_Y_actor.append(actors[1].cpu().numpy())
                    eval_Y_actor.append(actors[0].cpu().numpy())
                    # Initial Privacy
                    eval_Y_initial_actor.append(actors[0].cpu().numpy())
                    eval_Y_initial_actor.append(actors[1].cpu().numpy())

    # Print loss/accuracy
    print(f'--------------------\nEpoch {cur_tot_epoch+1}/{total_epochs}\n--------------------')
    cur_tot_epoch += 1

    if train_ae or train_cross:
        print(f'Training Loss:\t\t\t{np.mean(losses)}')
        if run_eval: print(f'Validation Loss:\t\t{np.mean(val_losses)}')
        print('\nTraining Losses:')
        print(f'Reconstruction Loss:\t\t{np.mean(rec_loss)}\nCross Reconstruction Loss:\t{np.mean(cross_loss)}\nEnd Effector Loss:\t\t{np.mean(end_effector_loss)}\nSmoothing Loss:\t\t\t{np.mean(smoothing_loss)}\nTriplet Loss:\t\t\t{np.mean(triplet_loss)}\nLatent Consistency Loss:\t{np.mean(latent_consistency_loss)}')
        if use_emb_adv:
            print(f'Privacy Loss:\t\t\t{np.mean(privacy_loss)}\nPrivacy Loss Dyn:\t\t{np.mean(privacy_loss_adv)}\nPrivacy Loss Stat:\t\t{np.mean(privacy_loss_coop)}')
            print(f'Utility Loss:\t\t\t{np.mean(utility_loss)}\nUtility Loss Dyn:\t\t{np.mean(utility_loss_adv)}\nUtility Loss Stat:\t\t{np.mean(utility_loss_coop)}')
        if use_discrim_adv: print(f'Discriminator Loss:\t\t{np.mean(discriminator_loss)}')

    if run_eval:
        print('\nValidation Losses:')
        print(f'Val Reconstruction Loss:\t{np.mean(val_rec_loss)}\nVal Cross Reconstruction Loss:\t{np.mean(val_cross_loss)}\nVal End Effector Loss:\t\t{np.mean(val_end_effector_loss)}\nVal Smoothing Loss:\t\t{np.mean(val_smoothing_loss)}\nVal Triplet Loss:\t\t{np.mean(val_triplet_loss)}\nVal Latent Consistency Loss:\t{np.mean(val_latent_consistency_loss)}')
        if use_emb_adv:
            print(f'Val Privacy Loss:\t\t{np.mean(val_privacy_loss)}\nVal Privacy Loss Dyn:\t\t{np.mean(val_privacy_loss_adv)}\nVal Privacy Loss Stat:\t\t{np.mean(val_privacy_loss_coop)}')
            print(f'Val Utility Loss:\t\t{np.mean(val_utility_loss)}\nVal Utility Loss Dyn:\t\t{np.mean(val_utility_loss_adv)}\nVal Utility Loss Stat:\t\t{np.mean(val_utility_loss_coop)}')
        if use_discrim_adv: print(f'Val Discriminator Loss:\t\t{np.mean(val_discriminator_loss)}')

    if train_emb_adv or train_discrim:
        print('\nEmbedding Classifers')
        if train_emb_adv and train_emb_this_epoch:
            print(f'Adv Privacy Training Loss:\t\t{np.mean(priv_training_loss)}\nAdv Utility Training Loss:\t\t{np.mean(util_training_loss)}\nCoop Privacy Training Loss:\t{np.mean(priv_coop_training_loss)}\nCoop Utility Training Loss:\t{np.mean(util_coop_training_loss)}\nDiscriminator Training Loss:\t{np.mean(discriminator_train_losses)}')
            print(f'Adv Privacy Training Acc:\t\t{np.mean(priv_training_acc)}\nAdv Utility Training Acc:\t\t{np.mean(util_training_acc)}\nCoop Privacy Training Acc:\t\t{np.mean(priv_coop_training_acc)}\nCoop Utility Training Acc:\t\t{np.mean(util_coop_training_acc)}\nDiscriminator Training Acc:\t{np.mean(discriminator_training_acc)}')
            if train_ae or train_cross: print(f'Privacy Acc Adv:\t\t{np.mean(privacy_acc_adv)}\nPrivacy Acc Coop:\t\t{np.mean(privacy_acc_coop)}\nUtility Acc Adv:\t\t{np.mean(utility_acc_adv)}\nUtility Acc Coop:\t\t{np.mean(utility_acc_coop)}')
            if run_eval: print(f'Val Privacy Acc Adv:\t\t{np.mean(val_privacy_acc_adv)}\nVal Privacy Acc Coop:\t\t{np.mean(val_privacy_acc_coop)}\nVal Utility Acc Adv:\t\t{np.mean(val_utility_acc_adv)}\nVal Utility Acc Coop:\t\t{np.mean(val_utility_acc_coop)}')

    if train_ae or train_cross: print(f'Discriminator Acc:\t\t{np.mean(discriminator_training_acc)}')
    if run_eval: print(f'Val Discriminator Acc:\t\t{np.mean(val_discriminator_acc)}')

    # Test Accuracy
    if run_sgn_eval and run_eval:
        print('\n')
        sgn_acc_known_acc, sgn_acc_known_f1, sgn_acc_known_prec, sgn_acc_known_recall, sgn_acc_known_topk = sgn_eval(eval_X_known, eval_Y_known_action, 'Known Action', is_action=True, k=k)
        sgn_acc_rec_acc, sgn_acc_rec_f1, sgn_acc_rec_prec, sgn_acc_rec_recall, sgn_acc_rec_topk = sgn_eval(eval_X_rec, eval_Y_rec_action, 'Reconstructed Action', is_action=True, k=k)
        sgn_acc_cross_acc, sgn_acc_cross_f1, sgn_acc_cross_prec, sgn_acc_cross_recall, sgn_acc_cross_topk = sgn_eval(eval_X, eval_Y_action, 'Generated Action', is_action=True, k=k)
        print('\n')
        sgn_priv_known_acc, sgn_priv_known_f1, sgn_priv_known_prec, sgn_priv_known_recall, sgn_priv_known_topk = sgn_eval(eval_X_known, eval_Y_known_actor, 'Known Actor', is_actor=True, k=k)
        sgn_priv_rec_acc, sgn_priv_rec_f1, sgn_priv_rec_prec, sgn_priv_rec_recall, sgn_priv_rec_topk = sgn_eval(eval_X_rec, eval_Y_rec_actor, 'Reconstructed Actor', is_actor=True, k=k)
        sgn_priv_cross_acc, sgn_priv_cross_f1, sgn_priv_cross_prec, sgn_priv_cross_recall, sgn_priv_cross_topk = sgn_eval(eval_X, eval_Y_actor, 'Generated Actor', is_actor=True, k=k)
        sgn_priv_initial_acc, sgn_priv_initial_f1, sgn_priv_initial_prec, sgn_priv_initial_recall, sgn_priv_initial_topk = sgn_eval(eval_X, eval_Y_initial_actor, 'Initial Actor', is_actor=True, k=k)
    else: print('\n')

    # Return dict with all losses and accuracies for plotting
    losses_dict = {}
    if train_ae or train_cross:
        losses_dict['loss'] = np.mean(losses)
        if run_eval: losses_dict['val_loss'] = np.mean(val_losses)
        losses_dict['rec_loss'] = np.mean(rec_loss)
        losses_dict['cross_loss'] = np.mean(cross_loss)
        losses_dict['end_effector_loss'] = np.mean(end_effector_loss)
        losses_dict['smoothing_loss'] = np.mean(smoothing_loss)
        losses_dict['triplet_loss'] = np.mean(triplet_loss)
        losses_dict['latent_consistency_loss'] = np.mean(latent_consistency_loss)
        losses_dict['privacy_loss'] = np.mean(privacy_loss)
        losses_dict['privacy_loss_adv'] = np.mean(privacy_loss_adv)
        losses_dict['privacy_loss_coop'] = np.mean(privacy_loss_coop)
        losses_dict['utility_loss'] = np.mean(utility_loss)
        losses_dict['utility_loss_adv'] = np.mean(utility_loss_adv)
        losses_dict['utility_loss_coop'] = np.mean(utility_loss_coop)
        losses_dict['discriminator_loss'] = np.mean(discriminator_loss)
    if run_eval:
        losses_dict['val_rec_loss'] = np.mean(val_rec_loss)
        losses_dict['val_cross_loss'] = np.mean(val_cross_loss)
        losses_dict['val_end_effector_loss'] = np.mean(val_end_effector_loss)
        losses_dict['val_smoothing_loss'] = np.mean(val_smoothing_loss)
        losses_dict['val_triplet_loss'] = np.mean(val_triplet_loss)
        losses_dict['val_latent_consistency_loss'] = np.mean(val_latent_consistency_loss)
        losses_dict['val_privacy_loss'] = np.mean(val_privacy_loss)
        losses_dict['val_privacy_loss_adv'] = np.mean(val_privacy_loss_adv)
        losses_dict['val_privacy_loss_coop'] = np.mean(val_privacy_loss_coop)
        losses_dict['val_utility_loss'] = np.mean(val_utility_loss)
        losses_dict['val_utility_loss_adv'] = np.mean(val_utility_loss_adv)
        losses_dict['val_utility_loss_coop'] = np.mean(val_utility_loss_coop)
        losses_dict['val_discriminator_loss'] = np.mean(val_discriminator_loss)
    if (train_emb_adv or train_discrim) and train_emb_this_epoch:
        losses_dict['priv_training_loss'] = np.mean(priv_training_loss)
        losses_dict['util_training_loss'] = np.mean(util_training_loss)
        losses_dict['discriminator_train_loss'] = np.mean(discriminator_train_losses)
        losses_dict['priv_training_acc'] = np.mean(priv_training_acc)
        losses_dict['util_training_acc'] = np.mean(util_training_acc)
        losses_dict['priv_coop_training_loss'] = np.mean(priv_coop_training_loss)
        losses_dict['priv_coop_training_acc'] = np.mean(priv_coop_training_acc)
        losses_dict['util_coop_training_loss'] = np.mean(util_coop_training_loss)
        losses_dict['util_coop_training_acc'] = np.mean(util_coop_training_acc)
        losses_dict['discriminator_training_acc'] = np.mean(discriminator_training_acc)
        if train_ae or train_cross:
            losses_dict['privacy_acc_adv'] = np.mean(privacy_acc_adv)
            losses_dict['privacy_acc_coop'] = np.mean(privacy_acc_coop)
            losses_dict['utility_acc_adv'] = np.mean(utility_acc_adv)
            losses_dict['utility_acc_coop'] = np.mean(utility_acc_coop)
        if run_eval:
            losses_dict['val_privacy_acc_adv'] = np.mean(val_privacy_acc_adv)
            losses_dict['val_privacy_acc_coop'] = np.mean(val_privacy_acc_coop)
            losses_dict['val_utility_acc_adv'] = np.mean(val_utility_acc_adv)
            losses_dict['val_utility_acc_coop'] = np.mean(val_utility_acc_coop)
    if train_ae or train_cross:
        losses_dict['discriminator_acc'] = np.mean(discriminator_training_acc)
    if run_eval:
        losses_dict['val_discriminator_acc'] = np.mean(val_discriminator_acc)
    if run_sgn_eval and run_eval:
        losses_dict['sgn_acc_known_acc'] = sgn_acc_known_acc
        losses_dict['sgn_acc_known_f1'] = sgn_acc_known_f1
        losses_dict['sgn_acc_known_prec'] = sgn_acc_known_prec
        losses_dict['sgn_acc_known_recall'] = sgn_acc_known_recall
        losses_dict['sgn_acc_rec_acc'] = sgn_acc_rec_acc
        losses_dict['sgn_acc_rec_f1'] = sgn_acc_rec_f1
        losses_dict['sgn_acc_rec_prec'] = sgn_acc_rec_prec
        losses_dict['sgn_acc_rec_recall'] = sgn_acc_rec_recall
        losses_dict['sgn_acc_cross_acc'] = sgn_acc_cross_acc
        losses_dict['sgn_acc_cross_f1'] = sgn_acc_cross_f1
        losses_dict['sgn_acc_cross_prec'] = sgn_acc_cross_prec
        losses_dict['sgn_acc_cross_recall'] = sgn_acc_cross_recall
        losses_dict['sgn_priv_known_acc'] = sgn_priv_known_acc
        losses_dict['sgn_priv_known_f1'] = sgn_priv_known_f1
        losses_dict['sgn_priv_known_prec'] = sgn_priv_known_prec
        losses_dict['sgn_priv_known_recall'] = sgn_priv_known_recall
        losses_dict['sgn_priv_rec_acc'] = sgn_priv_rec_acc
        losses_dict['sgn_priv_rec_f1'] = sgn_priv_rec_f1
        losses_dict['sgn_priv_rec_prec'] = sgn_priv_rec_prec
        losses_dict['sgn_priv_rec_recall'] = sgn_priv_rec_recall
        losses_dict['sgn_priv_cross_acc'] = sgn_priv_cross_acc
        losses_dict['sgn_priv_cross_f1'] = sgn_priv_cross_f1
        losses_dict['sgn_priv_cross_prec'] = sgn_priv_cross_prec
        losses_dict['sgn_priv_cross_recall'] = sgn_priv_cross_recall
        losses_dict['sgn_acc_known_topk'] = sgn_acc_known_topk
        losses_dict['sgn_acc_rec_topk'] = sgn_acc_rec_topk
        losses_dict['sgn_acc_cross_topk'] = sgn_acc_cross_topk
        losses_dict['sgn_priv_known_topk'] = sgn_priv_known_topk
        losses_dict['sgn_priv_rec_topk'] = sgn_priv_rec_topk
        losses_dict['sgn_priv_cross_topk'] = sgn_priv_cross_topk
        losses_dict['sgn_priv_initial_acc'] = sgn_priv_initial_acc
        losses_dict['sgn_priv_initial_f1'] = sgn_priv_initial_f1
        losses_dict['sgn_priv_initial_prec'] = sgn_priv_initial_prec
        losses_dict['sgn_priv_initial_recall'] = sgn_priv_initial_recall
        losses_dict['sgn_priv_initial_topk'] = sgn_priv_initial_topk

    # Save model
    if save and metric in losses_dict and losses_dict[metric] > 0:
        if matric_minimize:
            if np.mean(val_losses) < best_metric:
                best_metric = np.mean(val_losses)
                try:
                    # Move model to CPU before saving to avoid CUDA errors
                    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                    torch.save(cpu_state_dict, 'pretrained/MR.pt')
                except Exception as e:
                    print(f"Error saving model state dict: {e}")
        elif np.mean(val_losses) > best_metric:
            best_metric = np.mean(val_losses)
            try:
                # Move model to CPU before saving to avoid CUDA errors
                cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save(cpu_state_dict, 'pretrained/MR.pt')
            except Exception as e:
                print(f"Error saving model state dict: {e}")

    return losses_dict


def sgn_eval(X, Y, label='Undefined', is_actor=False, is_action=False, k=3):
    assert is_actor != is_action, "is_actor and is_action cannot both be True"
    assert is_actor or is_action, "Either is_actor or is_action must be True"

    if is_actor:
        classes = privacy_classes
        sgn = sgn_priv
    elif is_action:
        classes = utility_classes
        sgn = sgn_ar

    X = np.concatenate(X)
    X = np.pad(X, ((0,0), (0,225), (0,75)), 'constant')

    Y = np.concatenate(Y) - 1
    Y = np.eye(classes)[Y.astype(int)]

    acc, f1, prec, recall, topk = run_sgn_eval(sgn_train_x, sgn_train_y, X, Y, sgn_val_x, sgn_val_y, 1, sgn, k=k)
    print(f'\n{label} Accuracy:\t\t{acc}\n{label} F1:\t\t\t{f1*100}\n{label} Precision:\t\t{prec*100}\n{label} Recall:\t\t{recall*100}\n{label} Top-{k} Accuracy:\t{topk}\n')
    return acc, f1, prec, recall, topk

# Simplified training loop for only AE
def train_unpaired(run_eval=True, run_sgn_eval=True, save=True, ae=True, ee=False, triplet=False, use_emb_adv=False, use_discrim_adv=False, emb_adv=False, discrim_adv=False, k=3, smoothing=True):
    global best_metric
    global total_epochs
    global cur_tot_epoch

    # Store eval values for validation
    eval_X_known, eval_Y_known_action, eval_Y_known_actor, eval_X_rec, eval_Y_rec_action, eval_Y_rec_actor = [], [], [], [], [], []

    # Losses for printing
    rec_loss, end_effector_loss, smoothing_loss, triplet_loss, privacy_loss, privacy_loss_adv, privacy_loss_coop, privacy_acc_adv, privacy_acc_coop, priv_training_loss, utility_loss, utility_loss_adv, utility_loss_coop, utility_acc_adv, utility_acc_coop, util_training_loss, discriminator_loss, discriminator_train_losses, discriminator_acc, discriminator_train_accs, priv_coop_training_loss, priv_training_acc, priv_coop_training_acc, util_coop_training_loss, util_training_acc, util_coop_training_acc = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    val_rec_loss, val_end_effector_loss, val_smoothing_loss, val_triplet_loss, val_privacy_loss, val_privacy_loss_adv, val_privacy_loss_coop, val_privacy_acc_adv, val_privacy_acc_coop, val_utility_loss, val_utility_loss_adv, val_utility_loss_coop, val_utility_acc_adv, val_utility_acc_coop, val_discriminator_loss, val_discriminator_acc = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    losses, val_losses = [], []

    # Determine if adversaries need to be trained
    train_emb_this_epoch = True
    if emb_clf_update_per_epoch_unpaired < 1 and ae:
        if cur_tot_epoch % round(1 / emb_clf_update_per_epoch_unpaired) != 0:
            train_emb_this_epoch = False

    for (x, actors, actions) in rec_train_dl:
        # Move tensors to the configured device
        x = x.float().to(device)

        # Split into position and rotation
        if only_use_pos:
            x_pos = x
            x_rot = x
        else:
            x_pos = x[:, :, :, :3]
            x_rot = x[:, :, :, 3:]

        # Train adversaries
        if emb_adv or discrim_adv:
            # Train the discriminator
            if train_emb_this_epoch:
                it = 1
                if emb_clf_update_per_epoch_unpaired > 1: it = emb_clf_update_per_epoch_unpaired
                for _ in range(int(it)):
                    priv_train_loss, priv_train_coop_loss, util_train_loss, util_train_coop_loss, discriminator_train_loss, priv_acc, util_acc, priv_coop_acc, util_coop_acc, discriminator_train_acc = model.train_adv_unpaired(x_pos, x_rot, actors, actions, train_emb=emb_adv, train_discrim=discrim_adv)

                # Track the loss
                priv_training_loss.append(priv_train_loss)
                priv_coop_training_loss.append(priv_train_coop_loss)
                priv_training_acc.append(priv_acc)
                priv_coop_training_acc.append(priv_coop_acc)
                util_training_loss.append(util_train_loss)
                util_coop_training_loss.append(util_train_coop_loss)
                util_training_acc.append(util_acc)
                util_coop_training_acc.append(util_coop_acc)
                discriminator_train_losses.append(discriminator_train_loss)
                discriminator_train_accs.append(discriminator_train_acc)

        if not ae: continue

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        loss, _, losses_ = model.loss_unpaired(x_pos, x_rot, actors, actions, reconstruction=ae, emb_adv=use_emb_adv, discrim_adv=use_discrim_adv, ee=ee, triplet=triplet)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Track the loss
        losses.append(loss.item())

        rec_loss.append(losses_['rec_loss'])
        end_effector_loss.append(losses_['end_effector_loss'])
        smoothing_loss.append(losses_['smoothing_loss'])
        triplet_loss.append(losses_['triplet_loss'])
        privacy_loss.append(losses_['privacy_loss'])
        privacy_loss_adv.append(losses_['privacy_loss_adv'])
        privacy_loss_coop.append(losses_['privacy_loss_coop'])
        privacy_acc_adv.append(losses_['privacy_acc_adv'])
        privacy_acc_coop.append(losses_['privacy_acc_coop'])
        utility_loss.append(losses_['utility_loss'])
        utility_loss_adv.append(losses_['utility_loss_adv'])
        utility_loss_coop.append(losses_['utility_loss_coop'])
        utility_acc_adv.append(losses_['utility_acc_adv'])
        utility_acc_coop.append(losses_['utility_acc_coop'])
        discriminator_loss.append(losses_['discriminator_loss'])
        discriminator_acc.append(losses_['discriminator_acc'])


    # Decay learning rate
    # scheduler.step()

    # Validation
    if run_eval:
        with torch.no_grad():
            for (x, actors, actions) in rec_val_dl:
                x = x.float().to(device)

                # Split into position and rotation
                if only_use_pos:
                    x_pos = x
                    x_rot = x
                else:
                    x_pos = x[:, :, :, :3]
                    x_rot = x[:, :, :, 3:]

                loss, _, losses_ = model.loss_unpaired(x_pos, x_rot, actors, actions, reconstruction=ae, emb_adv=use_emb_adv, discrim_adv=use_discrim_adv, ee=ee, triplet=triplet)
                val_losses.append(loss.item())

                if run_sgn_eval:
                    eval_X_known.append(x_pos.contiguous().view(x_pos.size(0), T, -1).cpu().numpy())
                    eval_Y_known_action.append(np.array(actions))
                    eval_Y_known_actor.append(np.array(actors))

                    eval_X_rec.append(model(x_pos, x_rot).cpu().numpy())
                    eval_Y_rec_action.append(np.array(actions))
                    eval_Y_rec_actor.append(np.array(actors))

                val_rec_loss.append(losses_['rec_loss'])
                val_end_effector_loss.append(losses_['end_effector_loss'])
                val_smoothing_loss.append(losses_['smoothing_loss'])
                val_triplet_loss.append(losses_['triplet_loss'])
                val_privacy_loss.append(losses_['privacy_loss'])
                val_privacy_loss_adv.append(losses_['privacy_loss_adv'])
                val_privacy_loss_coop.append(losses_['privacy_loss_coop'])
                val_privacy_acc_adv.append(losses_['privacy_acc_adv'])
                val_privacy_acc_coop.append(losses_['privacy_acc_coop'])
                val_utility_loss.append(losses_['utility_loss'])
                val_utility_loss_adv.append(losses_['utility_loss_adv'])
                val_utility_loss_coop.append(losses_['utility_loss_coop'])
                val_utility_acc_adv.append(losses_['utility_acc_adv'])
                val_utility_acc_coop.append(losses_['utility_acc_coop'])
                val_discriminator_loss.append(losses_['discriminator_loss'])
                val_discriminator_acc.append(losses_['discriminator_acc'])

    # Print loss/accuracy
    print(f'--------------------\nEpoch {cur_tot_epoch+1}/{total_epochs}\n--------------------')
    cur_tot_epoch += 1
    if ae:
        print(f'Training Loss:\t\t\t{np.mean(losses)}\nValidation Loss:\t\t{np.mean(val_losses)}\n')
        print('Training Losses:')
        print(f'Reconstruction Loss:\t\t{np.mean(rec_loss)}\nEnd Effector Loss:\t\t{np.mean(end_effector_loss)}\nSmoothing Loss:\t\t\t{np.mean(smoothing_loss)}\nTriplet Loss:\t\t\t{np.mean(triplet_loss)}')
        if use_emb_adv:
            print(f'Privacy Loss:\t\t\t{np.mean(privacy_loss)}\nPrivacy Loss Dyn:\t\t{np.mean(privacy_loss_adv)}\nPrivacy Loss Stat:\t\t{np.mean(privacy_loss_coop)}')
            print(f'Utility Loss:\t\t\t{np.mean(utility_loss)}\nUtility Loss Dyn:\t\t{np.mean(utility_loss_adv)}\nUtility Loss Stat:\t\t{np.mean(utility_loss_coop)}')
        if use_discrim_adv: print(f'Discriminator Loss:\t\t{np.mean(discriminator_loss)}')
    if run_eval:
        print('\nValidation Losses:')
        print(f'Val Reconstruction Loss:\t{np.mean(val_rec_loss)}\nVal End Effector Loss:\t\t{np.mean(val_end_effector_loss)}\nVal Smoothing Loss:\t\t{np.mean(val_smoothing_loss)}\nVal Triplet Loss:\t\t{np.mean(val_triplet_loss)}')
        if use_emb_adv:
            print(f'Val Privacy Loss:\t\t{np.mean(val_privacy_loss)}\nVal Privacy Loss Dyn:\t\t{np.mean(val_privacy_loss_adv)}\nVal Privacy Loss Stat:\t\t{np.mean(val_privacy_loss_coop)}')
            print(f'Val Utility Loss:\t\t{np.mean(val_utility_loss)}\nVal Utility Loss Dyn:\t\t{np.mean(val_utility_loss_adv)}\nVal Utility Loss Stat:\t\t{np.mean(val_utility_loss_coop)}')
        if use_discrim_adv: print(f'Val Discriminator Loss:\t\t{np.mean(val_discriminator_loss)}')
    if (emb_adv or discrim_adv) and train_emb_this_epoch:
        print('\nAdversary Losses')
        print(f'Privacy Training Loss:\t\t{np.mean(priv_training_loss)}\nUtility Training Loss:\t\t{np.mean(util_training_loss)}\nDiscriminator Training Loss:\t{np.mean(discriminator_train_losses)}')
        print(f'Privacy Training Acc:\t\t{np.mean(priv_training_acc)}\nUtility Training Acc:\t\t{np.mean(util_training_acc)}\nDiscriminator Training Acc:\t{np.mean(discriminator_train_accs)}')
        print(f'Privacy Training Coop Loss:\t{np.mean(priv_coop_training_loss)}\nUtility Training Coop Loss:\t{np.mean(util_coop_training_loss)}')
        print(f'Privacy Training Coop Acc:\t{np.mean(priv_coop_training_acc)}\nUtility Training Coop Acc:\t{np.mean(util_coop_training_acc)}')
        if emb_adv and ae:
            print(f'Privacy Acc Adv:\t\t{np.mean(privacy_acc_adv)}\nPrivacy Acc Coop:\t\t{np.mean(privacy_acc_coop)}\nUtility Acc Adv:\t\t{np.mean(utility_acc_adv)}\nUtility Acc Coop:\t\t{np.mean(utility_acc_coop)}')
            if run_eval: print(f'Val Privacy Acc Adv:\t\t{np.mean(val_privacy_acc_adv)}\nVal Privacy Acc Coop:\t\t{np.mean(val_privacy_acc_coop)}\nVal Utility Acc Adv:\t\t{np.mean(val_utility_acc_adv)}\nVal Utility Acc Coop:\t\t{np.mean(val_utility_acc_coop)}')
        if discrim_adv and ae:
            print(f'Discriminator Acc:\t\t{np.mean(discriminator_acc)}')
            if run_eval: print(f'Val Discriminator Acc:\t\t{np.mean(val_discriminator_acc)}')


    # Test Accuracy
    if run_sgn_eval and run_eval:
        print('\n')
        sgn_acc_known_acc, sgn_acc_known_f1, sgn_acc_known_prec, sgn_acc_known_recall, sgn_acc_known_topk = sgn_eval(eval_X_known, eval_Y_known_action, 'Known Action', is_action=True, k=k)
        sgn_acc_rec_acc, sgn_acc_rec_f1, sgn_acc_rec_prec, sgn_acc_rec_recall, sgn_acc_rec_topk = sgn_eval(eval_X_rec, eval_Y_rec_action, 'Reconstructed Action', is_action=True, k=k)
        print('\n')
        sgn_priv_known_acc, sgn_priv_known_f1, sgn_priv_known_prec, sgn_priv_known_recall, sgn_priv_known_topk = sgn_eval(eval_X_known, eval_Y_known_actor, 'Known Actor', is_actor=True, k=k)
        sgn_priv_rec_acc, sgn_priv_rec_f1, sgn_priv_rec_prec, sgn_priv_rec_recall, sgn_priv_rec_topk = sgn_eval(eval_X_rec, eval_Y_rec_actor, 'Reconstructed Actor', is_actor=True, k=k)
        print('\n')
    else: print('\n')

    losses_dict = {}
    losses_dict['loss'] = np.mean(losses)

    if ae: losses_dict['rec_loss'] = np.mean(rec_loss)
    if ee: losses_dict['end_effector_loss'] = np.mean(end_effector_loss)
    if smoothing: losses_dict['smoothing_loss'] = np.mean(smoothing_loss)
    if triplet: losses_dict['triplet_loss'] = np.mean(triplet_loss)
    if run_eval:
        losses_dict['val_loss'] = np.mean(val_losses)
        if ae: losses_dict['val_rec_loss'] = np.mean(val_rec_loss)
        if ee: losses_dict['val_end_effector_loss'] = np.mean(val_end_effector_loss)
        if smoothing: losses_dict['val_smoothing_loss'] = np.mean(val_smoothing_loss)
        if triplet: losses_dict['val_triplet_loss'] = np.mean(val_triplet_loss)
        if run_sgn_eval:
            losses_dict['sgn_acc_known_acc'] = sgn_acc_known_acc
            losses_dict['sgn_acc_known_f1'] = sgn_acc_known_f1
            losses_dict['sgn_acc_known_prec'] = sgn_acc_known_prec
            losses_dict['sgn_acc_known_recall'] = sgn_acc_known_recall
            losses_dict['sgn_acc_rec_acc'] = sgn_acc_rec_acc
            losses_dict['sgn_acc_rec_f1'] = sgn_acc_rec_f1
            losses_dict['sgn_acc_rec_prec'] = sgn_acc_rec_prec
            losses_dict['sgn_acc_rec_recall'] = sgn_acc_rec_recall
            losses_dict['sgn_acc_known_topk'] = sgn_acc_known_topk
            losses_dict['sgn_acc_rec_topk'] = sgn_acc_rec_topk
            losses_dict['sgn_priv_known_acc'] = sgn_priv_known_acc
            losses_dict['sgn_priv_known_f1'] = sgn_priv_known_f1
            losses_dict['sgn_priv_known_prec'] = sgn_priv_known_prec
            losses_dict['sgn_priv_known_recall'] = sgn_priv_known_recall
            losses_dict['sgn_priv_rec_acc'] = sgn_priv_rec_acc
            losses_dict['sgn_priv_rec_f1'] = sgn_priv_rec_f1
            losses_dict['sgn_priv_rec_prec'] = sgn_priv_rec_prec
            losses_dict['sgn_priv_rec_recall'] = sgn_priv_rec_recall
            losses_dict['sgn_priv_known_topk'] = sgn_priv_known_topk
            losses_dict['sgn_priv_rec_topk'] = sgn_priv_rec_topk
    if use_emb_adv:
        losses_dict['privacy_loss'] = np.mean(privacy_loss)
        losses_dict['privacy_loss_adv'] = np.mean(privacy_loss_adv)
        losses_dict['privacy_loss_coop'] = np.mean(privacy_loss_coop)
        losses_dict['utility_loss'] = np.mean(utility_loss)
        losses_dict['utility_loss_adv'] = np.mean(utility_loss_adv)
        losses_dict['utility_loss_coop'] = np.mean(utility_loss_coop)
        losses_dict['privacy_acc_adv'] = np.mean(privacy_acc_adv)
        losses_dict['privacy_acc_coop'] = np.mean(privacy_acc_coop)
        losses_dict['utility_acc_adv'] = np.mean(utility_acc_adv)
        losses_dict['utility_acc_coop'] = np.mean(utility_acc_coop)
        if run_eval:
            losses_dict['val_privacy_acc_adv'] = np.mean(val_privacy_acc_adv)
            losses_dict['val_privacy_acc_coop'] = np.mean(val_privacy_acc_coop)
            losses_dict['val_utility_acc_adv'] = np.mean(val_utility_acc_adv)
            losses_dict['val_utility_acc_coop'] = np.mean(val_utility_acc_coop)
            losses_dict['val_privacy_loss'] = np.mean(val_privacy_loss)
            losses_dict['val_privacy_loss_adv'] = np.mean(val_privacy_loss_adv)
            losses_dict['val_privacy_loss_coop'] = np.mean(val_privacy_loss_coop)
            losses_dict['val_utility_loss'] = np.mean(val_utility_loss)
            losses_dict['val_utility_loss_adv'] = np.mean(val_utility_loss_adv)
            losses_dict['val_utility_loss_coop'] = np.mean(val_utility_loss_coop)
    if emb_adv and train_emb_this_epoch:
        losses_dict['priv_training_loss'] = np.mean(priv_training_loss)
        losses_dict['priv_training_acc'] = np.mean(priv_training_acc)
        losses_dict['priv_coop_training_loss'] = np.mean(priv_coop_training_loss)
        losses_dict['priv_coop_training_acc'] = np.mean(priv_coop_training_acc)
        losses_dict['util_training_loss'] = np.mean(util_training_loss)
        losses_dict['util_training_acc'] = np.mean(util_training_acc)
        losses_dict['util_coop_training_loss'] = np.mean(util_coop_training_loss)
        losses_dict['util_coop_training_acc'] = np.mean(util_coop_training_acc)
    if use_discrim_adv:
        losses_dict['discriminator_loss'] = np.mean(discriminator_loss)
        losses_dict['discriminator_acc'] = np.mean(discriminator_acc)
        if run_eval:
            losses_dict['val_discriminator_loss'] = np.mean(val_discriminator_loss)
            losses_dict['val_discriminator_acc'] = np.mean(val_discriminator_acc)
    if discrim_adv and train_emb_this_epoch:
        losses_dict['discriminator_train_loss'] = np.mean(discriminator_train_losses)
        losses_dict['discriminator_train_acc'] = np.mean(discriminator_train_accs)

    # Save model
    if save and metric in losses_dict and losses_dict[metric] > 0:
        if matric_minimize:
            if np.mean(val_losses) < best_metric:
                best_metric = np.mean(val_losses)
                try:
                    # Move model to CPU before saving to avoid CUDA errors
                    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                    torch.save(cpu_state_dict, 'pretrained/MR.pt')
                except Exception as e:
                    print(f"Error saving model state dict: {e}")
        elif np.mean(val_losses) > best_metric:
            best_metric = np.mean(val_losses)
            try:
                # Move model to CPU before saving to avoid CUDA errors
                cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save(cpu_state_dict, 'pretrained/MR.pt')
            except Exception as e:
                print(f"Error saving model state dict: {e}")

    return losses_dict
training_stages = [
    # Pre-Train Cross to separate embeddings
    {'epochs': 5, 'paired': True, 'ae': True, 'ee': True, 'cross': True, 'triplet': True, 'train_emb_adv': False, 'train_discrim_adv': False, 'emb_adv': False, 'discrim_adv': False, 'eval': False, 'sgn_eval': False, 'save': False},

    # Pre-Train AE
    {'epochs': 20, 'paired': False, 'ae': True, 'ee': True, 'cross': False, 'triplet': True, 'train_emb_adv': False, 'train_discrim_adv': False, 'emb_adv': False, 'discrim_adv': False, 'eval': False, 'sgn_eval': False, 'save': False},

    # Pre-Train Adversaries (Paired)
    {'epochs': 20, 'paired': True, 'ae': False, 'ee': False, 'cross': False, 'triplet': False, 'train_emb_adv': True, 'train_discrim_adv': True, 'emb_adv': False, 'discrim_adv': False, 'eval': False, 'sgn_eval': False, 'save': False},

    # Pre-Train Adversaries
    {'epochs': 50, 'paired': False, 'ae': False, 'ee': False, 'cross': False, 'triplet': False, 'train_emb_adv': True, 'train_discrim_adv': True, 'emb_adv': False, 'discrim_adv': False, 'eval': False, 'sgn_eval': False, 'save': False},

    # Train AE and adversaries with adversary loss
    {'epochs': 100, 'paired': False, 'ae': True, 'ee': True, 'cross': False, 'triplet': True, 'train_emb_adv': True, 'train_discrim_adv': True, 'emb_adv': True, 'discrim_adv': True, 'eval': True, 'sgn_eval': True, 'save': True},

    # Paired Training (Crossing)
    {'epochs': 100, 'paired': True, 'ae': True, 'ee': True, 'cross': True, 'triplet': True, 'train_emb_adv': True, 'train_discrim_adv': True, 'emb_adv': True, 'discrim_adv': True, 'eval': True, 'sgn_eval': False, 'save': True},
]
sgn_stage = {'epochs': 1, 'paired': True, 'ae': False, 'ee': False, 'cross': False, 'triplet': False, 'train_emb_adv': False, 'train_discrim_adv': False, 'emb_adv': True, 'discrim_adv': True, 'eval': True, 'sgn_eval': True, 'save': False}
total_epochs = sum([stage['epochs'] for stage in training_stages])
cur_tot_epoch = 0
if sgn_eval_after_each_stage: total_epochs += len(training_stages)

# Load AE pretrained model
# uncomment training stages to use full training
# pre_trained = 'pretrained/20240510-110342/stage_4.pt'
# pre_trained = 'pretrained/20240405-130711/stage_5.pt'
# model.load_state_dict(torch.load(pre_trained))

# mlflow logging
try: mlflow.end_run()
except: pass
mlflow.start_run()
mlflow.log_param('total_epochs', total_epochs)
mlflow.log_param('batch_size', batch_size)
mlflow.log_param('learning_rate', lr)
mlflow.log_param('one_dimension_conv', one_dimension_conv)
mlflow.log_param('ntu120', ntu_120)
mlflow.log_param('train_equal_test', str(not seperate_train_test))
mlflow.log_param('only_use_pos', str(only_use_pos))
mlflow.log_param('encoded_channels', str(encoded_channels))
mlflow.log_param('cross_samples_train', cross_samples_train)
mlflow.log_param('cross_samples_test', cross_samples_test)
mlflow.log_param('T', T)
mlflow.log_params(model.get_loss_params())

# os.mkdir('training_stages_log')
training_stage_name = f'training_stages_log/stages{datetime.now().strftime("%Y%m%d-%H%M%S")}.json'
with open(training_stage_name, 'w') as f:
    json.dump(training_stages, f)
mlflow.log_artifact(training_stage_name)

stages_save_path = f'pretrained/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
os.mkdir(stages_save_path)

for i, stage in enumerate(training_stages):
    print('\nMoving to new stage')
    print(stage, '\n')
    if stage['save']: assert stage['eval'], 'Cannot save model without evaluating'
    for epoch in range(stage['epochs']):
        if stage['sgn_eval']:
            if validation_acc_freq > 0 and epoch % validation_acc_freq == 0: use_sgn = True
            else: use_sgn = False
        else: use_sgn = False
        if not stage['paired']:
            log_dict = train_unpaired(run_eval=stage['eval'], run_sgn_eval= use_sgn, save=stage['save'], ae=stage['ae'], ee=stage['ee'], triplet=stage['triplet'], use_emb_adv=stage['emb_adv'], use_discrim_adv=stage['discrim_adv'], emb_adv=stage['train_emb_adv'], discrim_adv=stage['train_discrim_adv'], k=k)
        else:
            log_dict = train_paired(train_ae=stage['ae'], train_cross=stage['cross'], train_discrim=stage['train_discrim_adv'], train_emb_adv=stage['train_emb_adv'], run_eval=stage['eval'], use_emb_adv=stage['emb_adv'], use_discrim_adv=stage['discrim_adv'], run_sgn_eval= use_sgn, save=stage['save'], k=k)

        for key, value in log_dict.items():
            mlflow.log_metric(key, value, step=cur_tot_epoch-1)

    # save model
    try:
        # Move model to CPU before saving to avoid CUDA errors
        cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state_dict, f'{stages_save_path}/stage_{i}.pt')
    except Exception as e:
        print(f"Error saving model state dict for stage {i}: {e}")

    if sgn_eval_after_each_stage:
        print('\nEvaluating Stage\n')
        stage = sgn_stage
        log_dict = train_paired(train_ae=stage['ae'], train_cross=stage['cross'], train_discrim=stage['train_discrim_adv'], train_emb_adv=stage['train_emb_adv'], run_eval=stage['eval'], use_emb_adv=stage['emb_adv'], use_discrim_adv=stage['discrim_adv'], run_sgn_eval= use_sgn, save=stage['save'], k=k)
        cur_tot_epoch += 1
        for key, value in log_dict.items():
            mlflow.log_metric(key, value, step=cur_tot_epoch-1)

try:
    # Move model to CPU before saving to avoid CUDA errors
    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    mlflow.pytorch.log_state_dict(cpu_state_dict, 'final_model')
except Exception as e:
    print(f"Error logging final model state dict: {e}")
mlflow.end_run()