#!/bin/bash
{
gnome-terminal -t "cam1_l" -x bash -c "python3 IPC1_Live.py"
}&
sleep 0.01s
{
gnome-terminal -t "cam2_l" -x bash -c "python3 IPC2_Live.py"
}&
sleep 0.01s
{
gnome-terminal -t "cam1_s" -x bash -c "./IPC1_Server_FFMpeg"

sleep 0.01s
{
gnome-terminal -t "cam2_s" -x bash -c "./IPC2_Server_FFMpeg"
}

