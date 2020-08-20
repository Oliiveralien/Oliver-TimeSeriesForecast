python main.py --hidSkip 10 --issue traffic --model CNN
python main.py --hidSkip 10 --issue traffic --model RNN
python main.py --hidSkip 10 --issue traffic --model MHA_Net
python main.py --hidSkip 10 --issue traffic --model LSTNet
python main.py --issue ele --horizon 6 --output_fun Linear --skip 0 --batch_size 32 --hidCNN 50 --hidRNN 50 --model CNN
python main.py --issue ele --horizon 6 --output_fun Linear --skip 0 --batch_size 32 --hidCNN 50 --hidRNN 50 --model RNN
python main.py --issue ele --horizon 6 --output_fun Linear --skip 0 --batch_size 32 --hidCNN 50 --hidRNN 50 --model MHA_Net
python main.py --issue ele --horizon 6 --output_fun Linear --skip 0 --batch_size 32 --hidCNN 50 --hidRNN 50 --model LSTNet
python main.py --issue solar --skip 0 --output_fun Linear --model CNN
python main.py --issue solar --skip 0 --output_fun Linear --model RNN
python main.py --issue solar --skip 0 --output_fun Linear --model MHA_Net
python main.py --issue solar --skip 0 --output_fun Linear --model LSTNet
python main.py --issue stock --hidCNN 50 --hidRNN 50 --L1Loss False --output_fun None --model CNN
python main.py --issue stock --hidCNN 50 --hidRNN 50 --L1Loss False --output_fun None --model RNN
python main.py --issue stock --hidCNN 50 --hidRNN 50 --L1Loss False --output_fun None --model MHA_Net
python main.py --issue stock --hidCNN 50 --hidRNN 50 --L1Loss False --output_fun None --model LSTNet