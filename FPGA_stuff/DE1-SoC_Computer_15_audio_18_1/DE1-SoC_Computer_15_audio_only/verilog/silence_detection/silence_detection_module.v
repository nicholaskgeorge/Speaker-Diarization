module silence_detection #(
    parameter FRAME_LENGTH = 4800
    parameter NUM_FSM_STATES = 4
    parameter AUDIO_DATA_BIT_SIZE = 32
    parameter ZCR_THRESHOLD = 
)
(
    input clk,
    input reset,
    
    input input_audio [AUDIO_DATA_BIT_SIZE-1:0],
    input dac_audio_valid,

    output val,
    output val_ready
)
    //states for silence detection FSM
    localparam [1:0] idle  = 2'b00,
                     store_prev_sign = 2'b01,
                     compar_signs = 2'b10,
                     silence_or_not  = 2'b11;

    //count # of samples analysed
    reg sample_counter [$clog2(FRAME_LENGTH)-1:0];
    reg prev_sign;
    reg current_state [$clog2(NUM_FSM_STATES)-1:0];
    reg next_state [$clog2(NUM_FSM_STATES)-1:0];
    reg sample_store [BUS_DATA_BIT_SIZE-1:0];
    //note: since two sample are needed to determine a crossing then
    //      then the num_crossings can be at max FRAME_LENGTH/2
    reg num_crossings [$clog2(FRAME_LENGTH/2)-1:0];
    reg zero_cross_count
    reg new_val;

    always @(posedge clk or posedge reset) begin
        if (reset)
            current_state <= idle;
            new_val = 0;
            sample_counter = 0;
        else
            current_state <= next_state;
    end

    always @(posedge) begin
        //on every cycle store the audio, only use the input if valid
        sample_store <= input_audio;

        if(current_state == idle) begin
            val_ready <= 0;
            if (dac_audio_valid)
                next_state <= store_prev_sign;
                new_val <= 1;
        end

        if(current_state == store_prev_sign) begin
            val_ready <= 0;
            if(new_val) begin
                prev_sign <= sample_store[31]; // get the sign of the sample
                new_val <= 0;
                sample_counter <= sample_counter+1;
            end
            if(dac_audio_valid)
                next_state <= compare_signs;
                new_val <= 1;
        end

        if(current_state == compare_signs) begin
            //if there is a new value and the sign is the same as the previous one
            if(new_val and (prev_sign != sample_store[31])) begin
                sample_counter <= sample_counter+1;
                num_crossings <= num_crossings+1;
                new_val <= 0;
                //if this crossing is the very last one
                if (sample_counter == FRAME_LENGTH-1)begin
                    next_state <= silence_or_not
                end

            //if there is new value and there was no crossing
            end else if (new_val and (prev_sign == sample_store[31])) begin
                sample_counter <= sample_counter+1;
                if(dac_audio_valid) begin
                    new_val <= 1;
                    current_state <= store_prev_sign;
                end
                else
                    new_val <= 0;

            //otherwise just wait until there is another valid sample from the DAC
            end else if (dac_audio_valid) begin
                new_val <= 1;
                current_state <= store_prev_sign;
            end
            
        end

        if(current_state == silence_or_not) begin
            val_ready <= 1;
            if(num_crossings <= ZCR_THRESHOLD)
                val <= 1;
            else 
                val <= 0;
            current_state <= idle;
        end
    end
    


endmodule