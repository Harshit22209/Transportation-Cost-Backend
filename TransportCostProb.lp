\* Transport_Cost_Prob *\
Minimize
Sum_of_Transporting_Costs: 2 Route_A_1 + 4 Route_A_2 + 5 Route_A_3
 + 2 Route_A_4 + 3 Route_B_1 + Route_B_2 + 3 Route_B_3 + 2 Route_B_4
Subject To
Sum_of_Products_into_Retail1: Route_A_1 + Route_B_1 >= 9784.02539062
Sum_of_Products_into_Retail2: Route_A_2 + Route_B_2 >= 14026.4667969
Sum_of_Products_into_Retail3: Route_A_3 + Route_B_3 >= 6467.59130859
Sum_of_Products_into_Retail4: Route_A_4 + Route_B_4 >= 3430.60180664
Sum_of_Products_out_of_Warehouse_A: Route_A_1 + Route_A_2 + Route_A_3
 + Route_A_4 <= 20000
Sum_of_Products_out_of_Warehouse_B: Route_B_1 + Route_B_2 + Route_B_3
 + Route_B_4 <= 18000
Bounds
 0 <= Route_A_1
 0 <= Route_A_2
 0 <= Route_A_3
 0 <= Route_A_4
 0 <= Route_B_1
 0 <= Route_B_2
 0 <= Route_B_3
 0 <= Route_B_4
Generals
Route_A_1
Route_A_2
Route_A_3
Route_A_4
Route_B_1
Route_B_2
Route_B_3
Route_B_4
End
