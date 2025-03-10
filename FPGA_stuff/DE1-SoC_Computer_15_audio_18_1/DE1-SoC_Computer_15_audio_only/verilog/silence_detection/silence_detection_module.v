module silence_detection #(
    parameter FRAME_LENGTH = 4800,
    parameter NUM_FSM_STATES = 4,
    parameter AUDIO_DATA_BIT_SIZE = 32,
    parameter ZCR_THRESHOLD = 80,
    parameter SHIFT_OFFSET = 6
)
(
    input clk,
    input reset,
    
    input [AUDIO_DATA_BIT_SIZE-1:0] input_audio,
    input dac_audio_valid,

    output reg val,
    output reg val_ready,

    //this signal goes high on the cycle the first sample  of the segment is analyzed
    //it goes low on the cycle the result is given
    output reg analyzing_segment
);
    //states for silence detection FSM
    localparam [1:0] idle                     = 2'b00,
                     first_sample_sign_stored = 2'b01,
                     compare_signs            = 2'b10,
                     silence_or_not           = 2'b11;
    
    localparam SAMPLE_COUNTER_SIZE = $clog2(FRAME_LENGTH);
    localparam STATE_REG_SIZE = $clog2(NUM_FSM_STATES);
    localparam NUM_CROSSINGS_REG_SIZE = $clog2(FRAME_LENGTH/2);
    localparam ZCR_REG_SIZE = $clog2(FRAME_LENGTH);


    //count # of samples analysed
    reg [SAMPLE_COUNTER_SIZE-1:0] sample_counter;
    reg sample1_sign;
    reg sample2_sign;
    reg [STATE_REG_SIZE-1:0] current_state;
    reg [STATE_REG_SIZE-1:0] next_state;
    //note: since two sample are needed to determine a crossing then
    //      then the num_crossings can be at max FRAME_LENGTH/2
    reg [NUM_CROSSINGS_REG_SIZE-1:0] num_crossings;

    wire [ZCR_REG_SIZE-1:0] zcr;
    //get zcr by normalizing by the length of the segment. shift offset so that
    //the comparisons do not need to be decimals
    assign zcr = num_crossings>>(ZCR_REG_SIZE-SHIFT_OFFSET);

    wire current_sample_sign;

    //the input signals are signed so just get the first bit to check sign
    assign current_sample_sign = input_audio[31];

    always @(posedge clk) begin
        current_state <= next_state;
        //always capture the audio

        if (reset) begin
            sample_counter <= 0;
            current_state  <= idle;
            num_crossings  <= 0;
            sample1_sign   <= 0;
            sample2_sign   <= 0;
        end
        if (current_state == idle)begin
            if (dac_audio_valid) begin
                //must catch the sign now BEFORE going to first_sample_sign_stored state
                sample1_sign <= current_sample_sign;
                //increment sample count
                sample_counter <= sample_counter + 1;
            end
            //infered latch fix
            num_crossings <= num_crossings;
        end
        if (current_state == first_sample_sign_stored)begin
            //store the sign BEFORE going to comparison
            if(dac_audio_valid) begin
                sample2_sign <= current_sample_sign;
                sample_counter <= sample_counter + 1;
            end
            num_crossings <= num_crossings;
        end
        if (current_state == compare_signs) begin
            //if have crossed count the number of crossings
            num_crossings <= num_crossings + (sample1_sign == sample2_sign);
            if (dac_audio_valid) begin
                sample1_sign <= current_sample_sign;
                sample_counter <= sample_counter+1;
            end else begin
                sample2_sign <= 0;
                sample_counter <= sample_counter;
            end

            num_crossings <= num_crossings;
        end
        if (current_state == silence_or_not)begin
            num_crossings <= 0;
            sample_counter <= 0;
            num_crossings <= num_crossings;
        end
    end
        



    always @(*) begin
        if (reset) begin
            val_ready = 0;
            analyzing_segment = 0;
            val = 0;
            next_state = idle;
        end
        else if (current_state == idle)begin
            val_ready = 0;
            analyzing_segment = 0;
            val = 0;
            if(dac_audio_valid)
                next_state = first_sample_sign_stored;
            else
                next_state = idle;
        end
        else if (current_state == first_sample_sign_stored)begin
            val_ready = 0;
            analyzing_segment = 1;
            val = 0;
            if(dac_audio_valid)
                next_state = compare_signs;
            else
                next_state = first_sample_sign_stored;
        end
        else if (current_state == compare_signs)begin
            val_ready = 0;
            analyzing_segment = 1;
            val = 0;
            if (sample_counter == FRAME_LENGTH)
                next_state = silence_or_not;
            else if (dac_audio_valid)
                next_state = first_sample_sign_stored;
            else
                next_state = compare_signs;
        end
        else if (current_state == silence_or_not)begin
            val_ready = 1;
            analyzing_segment = 0;
            next_state = idle;
            //silence present if num_crossings less than threshold
            val = (zcr <= ZCR_THRESHOLD);
        end
        else begin
            val_ready = 0;
            analyzing_segment = 0;
            val = 0;
            next_state = idle;
        end
    end
endmodule