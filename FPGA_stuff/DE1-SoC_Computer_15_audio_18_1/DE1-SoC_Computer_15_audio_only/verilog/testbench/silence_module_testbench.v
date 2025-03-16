`timescale 1ns/1ps

module silence_detection_tb;
    parameter FRAME_LENGTH = 15038;
    parameter AUDIO_DATA_BIT_SIZE = 32;
    parameter ZCR_THRESHOLD = 80;
    
    reg clk;
    reg reset;
    reg [AUDIO_DATA_BIT_SIZE-1:0] input_audio;
    reg dac_audio_valid;
    reg silent_or_not;
    
    wire val;
    wire val_ready;
    wire analyzing_segment;
    
    silence_detection #(
        .FRAME_LENGTH(FRAME_LENGTH),
        .AUDIO_DATA_BIT_SIZE(AUDIO_DATA_BIT_SIZE),
        .ZCR_THRESHOLD(ZCR_THRESHOLD)
    ) uut (
        .clk(clk),
        .reset(reset),
        .input_audio(input_audio),
        .dac_audio_valid(dac_audio_valid),
        .val(val),
        .val_ready(val_ready),
        .analyzing_segment(analyzing_segment)
    );
    reg [3:0] toggle_count;
    reg [15:0] sign_switch_count;

    //counts the number of axis crossings.
    wire [15:0] sign_switch_count_final;
    assign sign_switch_count_final = sign_switch_count>>1;
    
    // Clock generation
    always #5 clk = ~clk;
    
    task run_test;
        input integer test_length;
        begin
            // Initialize signals
            reset = 1;
            input_audio = 0;
            dac_audio_valid = 0;
            silent_or_not = 0;
            
            // Apply reset
            #20 reset = 0;
            
            // Generate alternating signal (1 and -1) on posedge of clk
            repeat (test_length*3) begin
                @(posedge clk);
                input_audio <= (input_audio == 32'h00000001) ? 32'hFFFFFFFF : 32'h00000001;
                dac_audio_valid <= 1;
            end
            
            @(posedge clk);
            dac_audio_valid <= 0;
            
            // Wait for processing
            #10;
            
            // Determine if silence was detected
            silent_or_not = (val_ready && val) ? 1 : 0;
            
            // Print result
            if (silent_or_not)
                $display("Silence detected.");
            else
                $display("No silence detected.");
        end
    endtask

    task random_delay_test;
        input integer test_length;
        integer random_value; // Moved declaration
        begin
            // Initialize signals
            reset = 1;
            input_audio = 0;
            dac_audio_valid = 0;
            silent_or_not = 0;
            sign_switch_count = 0;
            
            // Apply reset
            #20 reset = 0;
            
            // Generate alternating signal (1 and -1) on posedge of clk
            repeat (test_length*4) begin
                #10
                @(posedge clk);
                // Randomly switching the sign
                random_value = $urandom_range(1, 0); 
                $display("Random value between 1 and 0: %d", random_value);
                if (random_value == 1) begin
                    dac_audio_valid <= 1;
                    input_audio <= (input_audio == 32'h00000001) ? 32'hFFFFFFFF : 32'h00000001;
                    sign_switch_count <= sign_switch_count + 1;
                end else begin
                    dac_audio_valid <= 0;
                    input_audio <= input_audio;
                end
            end
            
            // Wait for processing
            #10;
            
            // Determine if silence was detected
            silent_or_not = (val_ready && val) ? 1 : 0;
            
            // Print result
            if (silent_or_not)
                $display("Silence detected.");
            else
                $display("No silence detected.");
        end
    endtask

    task random_delay_test_threshold_test;
        input integer test_length;
        input integer max_num_crossings;
        integer random_value; // Moved declaration
        
        begin
            // Initialize signals
            reset = 1;
            input_audio = 0;
            dac_audio_valid = 0;
            silent_or_not = 0;
            sign_switch_count = 0;
            
            // Apply reset
            #20 reset = 0;
            
            // Generate alternating signal (1 and -1) on posedge of clk
            repeat (test_length*4) begin
                #10
                @(posedge clk);
                // Randomly switching the sign
                random_value = 1;//$urandom_range(1, 0); 
                if (random_value == 1) begin
                    dac_audio_valid <= 1;
                    if(sign_switch_count_final <= max_num_crossings) begin
                        input_audio <= (input_audio == 32'h00000001) ? 32'hFFFFFFFF : 32'h00000001;
                        sign_switch_count <= sign_switch_count + 1;
                    end
                end else begin
                    dac_audio_valid <= 0;
                    input_audio <= input_audio;
                end
            end
            
            // Wait for processing
            #10;
            
            // Determine if silence was detected
            silent_or_not = (val_ready && val) ? 1 : 0;
            
            // Print result
            if (silent_or_not)
                $display("Silence detected.");
            else
                $display("No silence detected.");
        end
    endtask






    initial begin
        clk = 0;
        
        // Run first test case
        // run_test(FRAME_LENGTH);
        random_delay_test_threshold_test(FRAME_LENGTH, 8900);
        // Add more test cases here if needed
        #200 $stop;
    end
endmodule
