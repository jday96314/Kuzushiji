import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = torch.device("cuda")

class Encoder(nn.Module):
	def __init__(self, char_class_count, embedding_dim_count, recurrent_unit_count):
		super(Encoder, self).__init__()
		self.Embedding = nn.Embedding(char_class_count, embedding_dim_count)
		self.RecurrentUnitCount = recurrent_unit_count
		self.Gru = nn.GRU(
			input_size = embedding_dim_count + 2,
			hidden_size = self.RecurrentUnitCount,
			num_layers = 1,
			bidirectional = False)
		
	def forward(self, input_character_locations, input_character_labels, hidden_state):
		embedded_character_labels = self.Embedding(input_character_labels).view(
			input_character_labels.shape[0], input_character_labels.shape[1], -1)
		full_inputs = torch.cat([input_character_locations, embedded_character_labels], dim = 2)
		output, hidden_state = self.Gru(full_inputs, hidden_state)
		
		return output, hidden_state

	def initHidden(self, batch_size):
		return torch.zeros(1, batch_size, self.RecurrentUnitCount, device=DEVICE)

class Decoder(nn.Module):
	def __init__(self, recurrent_unit_count, embedding_dim_count, char_label_class_count, dropout_rate, max_sequence_length):
		super(Decoder, self).__init__()
		self.RecurrentUnitCount = recurrent_unit_count

		self.EmbeddingLayer = nn.Embedding(char_label_class_count, embedding_dim_count)
		self.AttentionLayer = nn.Linear(1*recurrent_unit_count + embedding_dim_count + 2, max_sequence_length)
		self.Combiner = nn.Linear(1*recurrent_unit_count + embedding_dim_count + 2, recurrent_unit_count)
		self.Dropout = nn.Dropout(dropout_rate)
		self.Gru = nn.GRU(
			input_size = recurrent_unit_count, 
			hidden_size = self.RecurrentUnitCount,
			num_layers = 1,
			bidirectional = False)

		self.CharLocationOutput = nn.Linear(1*recurrent_unit_count, 2)
		self.CharLabelOutput = nn.Linear(1*recurrent_unit_count, char_label_class_count)

	def forward(self, input_character_locations, input_character_labels, hidden_state, encoder_outputs):
		embedded_labels = self.EmbeddingLayer(input_character_labels).view(
			input_character_labels.shape[0], input_character_labels.shape[1], -1)
		embedded_labels = self.Dropout(embedded_labels)
		embedded_labels = torch.cat([input_character_locations.unsqueeze(1), embedded_labels], dim = 2)

		batch_size = len(input_character_labels)
		concatenated_label_and_state = torch.cat((
			embedded_labels.reshape(batch_size, -1), 
			hidden_state.transpose(0,1).reshape(batch_size, -1)), dim = 1)
		raw_attention_output = self.AttentionLayer(concatenated_label_and_state)[:,:len(encoder_outputs)]
		attention_weights = F.softmax(raw_attention_output, dim=1)
		attended_encoder_output = attention_weights.unsqueeze(1).bmm(encoder_outputs.transpose(0,1))

		gru_input = torch.cat(
			(embedded_labels.transpose(0,1)[0], attended_encoder_output.transpose(0,1)[0]), 
			1)
		gru_input = self.Combiner(gru_input).unsqueeze(0)
		gru_input = F.relu(gru_input)

		gru_output, hidden_state = self.Gru(gru_input, hidden_state)

		char_locations = self.CharLocationOutput(gru_output[0])
		char_labels = F.log_softmax(self.CharLabelOutput(gru_output[0]), dim=1)
		return char_locations, char_labels, hidden_state

	def initHidden(self, batch_size):
		return torch.zeros(1, batch_size, self.RecurrentUnitCount, device=DEVICE)
