python main-derain.py \
        --test_root=./test_data/ \
        --load_model_path_loca=./checkpoints_loca/Loca-epoch49.pth \
        --load_model_path_atm_light=./checkpoints_t_vapor_loss_all/rain/complex_rain/refine_atm_light-epoch9.pth \
        --load_model_path_transmission=./checkpoints_t_vapor_loss_all/rain/complex_rain/refine_transmission-epoch9.pth \
        --load_model_path_atm_light_before_tuned=./checkpoints_atm_light/rain/complex/atm_light-epoch13.pth \
        --load_model_path_transmission_vapor=./checkpoints_t_vapor_loss_all/rain/complex_rain/t_vapor-epoch9.pth \
	    --test_batch_size=1 \
        --cuda

