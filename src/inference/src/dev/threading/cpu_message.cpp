// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/runtime/threading/cpu_message.hpp"

#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <vector>

namespace ov {
namespace threading {

MessageManager::MessageManager() {
    _num_sub_streams = 0;
    call_back_count = 0;
}

MessageManager::~MessageManager() {}

void MessageManager::send_message(const MessageInfo& msg_info) {
    {
        std::lock_guard<std::mutex> lock(_msgMutex);
        _messageQueue.push_back(msg_info);
        // std::cout << "send : " << msg_info.msg_type << "\n";
    }
    _msgCondVar.notify_all();
}

void MessageManager::infer_wait() {
    std::unique_lock<std::mutex> lock(_inferMutex);
    _inferCondVar.wait(lock, [&] {
        return call_back_count.load() >= _num_sub_streams;
    });
    call_back_count = 0;
}

void MessageManager::server_wait() {
    if (!_serverThread.joinable()) {
        assert(_num_sub_streams);
        MsgType msg_type;
        _serverThread = std::thread([&]() {
            int count = 0;
            while (!_isServerStopped) {
                std::vector<MessageInfo> msgQueue;
                {
                    // std::cout << "server_wait ........" << _isServerStopped << "\n";
                    std::unique_lock<std::mutex> lock(_msgMutex);
                    _msgCondVar.wait(lock, [&] {
                        return !_messageQueue.empty();
                    });
                    std::swap(_messageQueue, msgQueue);
                    // std::cout << "server_wait receive: " << msgQueue[0].msg_type << "\n";
                }

                for (auto rec_info : msgQueue) {
                    msg_type = rec_info.msg_type;
                    if (msg_type == START_INFER) {
                        Task task = std::move(rec_info.task);
                        task();
                    } else if (msg_type == CALL_BACK) {  // CALL_BACK
                        count++;
                        // std::cout << "server_wait CALL_BACK: " << call_back_count << "/" << _num_sub_streams << "\n";
                        if (count == _num_sub_streams) {
                            call_back_count = count;
                            count = 0;
                            _inferCondVar.notify_one();
                        }
                    } else if (msg_type == QUIT) {
                        _isServerStopped = true;
                    }
                }
            }
            // std::cout << "-------- server_wait end ---------\n";
        });
    }
}

void MessageManager::set_num_sub_streams(int num_sub_streams) {
    _num_sub_streams = num_sub_streams;
}

int MessageManager::get_num_sub_streams() {
    return _num_sub_streams;
}

void MessageManager::stop_server_thread() {
    MessageInfo msg_info;
    msg_info.msg_type = ov::threading::MsgType::QUIT;
    send_message(msg_info);
    if (_serverThread.joinable()) {
        _serverThread.join();
    }
}

namespace {

class MessageManageHolder {
    std::mutex _mutex;
    std::weak_ptr<MessageManager> _manager;

public:
    MessageManageHolder(const MessageManageHolder&) = delete;
    MessageManageHolder& operator=(const MessageManageHolder&) = delete;

    MessageManageHolder() = default;

    std::shared_ptr<ov::threading::MessageManager> get() {
        std::lock_guard<std::mutex> lock(_mutex);
        auto manager = _manager.lock();
        if (!manager) {
            _manager = manager = std::make_shared<MessageManager>();
        }
        return manager;
    }
};

}  // namespace

std::shared_ptr<MessageManager> message_manager() {
    static MessageManageHolder message_manage;
    auto a = message_manage.get();
    return a;
}

}  // namespace threading
}  // namespace ov